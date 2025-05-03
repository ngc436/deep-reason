import ast
import logging
from dataclasses import dataclass
from typing import List, Optional, Any

from langchain_core.language_models import LanguageModelInput

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableParallel, RunnableLambda, Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryOutputParser
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def create_structured_chain(
        prompt: ChatPromptTemplate,
        llm: Runnable[PromptValue, BaseMessage],
        parser: PydanticOutputParser,
        max_parsing_retries = 3) -> Runnable:

    chain = (
        prompt
        | RunnableParallel(completion=llm, prompt_value=RunnablePassthrough())
    )

    retry_planner_parser = RetryOutputParser.from_llm(
        parser=parser,
        llm=llm,
        prompt=PromptTemplate.from_template("{prompt}"),
        max_retries=max_parsing_retries
    )
    
    def _do_parsing_retrying(x: dict):
        result = None
        completion = x['completion'].content
        prompt_value = x['prompt_value']
        try:
            result = retry_planner_parser.parse_with_prompt(completion=completion,prompt_value=prompt_value)
        except OutputParserException:
            logger.warning("Proceeding without result due to parser errors (even after retrying).")

        return result

    chain = (
        RunnableLambda(lambda x: {**x, "response_format_description": parser.get_format_instructions()})
        | chain
        | RunnableLambda(_do_parsing_retrying, name="retry_planner_lambda")
    )

    return chain


def _doc2str(doc: Document) -> str:
    return (f"Следующий фрагмент относится к файлу {doc.metadata.get('filename', '')} "
            f"и главе {doc.metadata.get('chapter', '')}."
            f"\n{doc.page_content}")


@dataclass
class _RerankedDocState:
    doc: Document
    ranking_prompt: PromptValue
    answer: Optional[str] = None
    score: Optional[float] = None


@dataclass
class _RerankedDocs:
    top_docs: List[_RerankedDocState]
    all_docs: List[_RerankedDocState]


class VLLMChatOpenAI(ChatOpenAI):
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # max_tokens was deprecated in favor of max_completion_tokens
        # in September 2024 release
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload



class PlannerOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        """Parse by splitting."""
        try:
            result = ast.literal_eval(text.split('ЗАПРОСЫ:')[
                                      1].strip().split(']')[0] + ']')
        except Exception as ex:
            logger.warning("Planner output parsing exception!", exc_info=True)
            raise OutputParserException(str(ex))
        return result
    

class MultiQueryRetriever:
    def __init__(self, retriever: List[VectorStoreRetriever] | VectorStoreRetriever):
        self._retrievers = [retriever] if isinstance(
            retriever, VectorStoreRetriever) else retriever

    @staticmethod
    def remove_duplicates(docs: List[Document]) -> List[Document]:
        return list({doc.page_content: doc for doc in docs}.values())

    def do_batch(self, queries: List[str]) -> List[Document]:
        result = [
            doc
            for retriever in self._retrievers
            for docs in retriever.batch(queries)
            for doc in docs
        ]
        result = self.remove_duplicates(result)
        return result

    async def do_abatch(self, arr: List[str]) -> List[Document]:
        result = [
            doc
            for retriever in self._retrievers
            for docs in (await retriever.abatch(arr))
            for doc in docs
        ]

        result = self.remove_duplicates(result)
        return result

