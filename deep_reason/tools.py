import logging
from langchain.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from abc import abstractmethod
from typing import Optional, Union, List, Tuple, Callable, Awaitable, Dict, Any
from deep_reason.utils import Input, StateT, ModelType, StreamQAPipeline, PipelineEvent
from langchain_core.messages import BaseMessage
from langchain_core.embeddings import Embeddings
from langfuse.callback import CallbackHandler
from deep_reason.schemes import WebIntermediateOutputs

logger = logging.getLogger(__name__)

class Tool:
    def __init__(self, stream_events: bool = True):
        self._stream_events = stream_events
        self.tool = StructuredTool.from_function(
            coroutine=self.invoke, name=self._tool_name, description=self._description)

    async def invoke(self, question: str, config: RunnableConfig, chat_history: Optional[Union[List[BaseMessage], List[Tuple[str, str]]]] = None) -> str:
        pipeline = await self.create_pipeline(config)
        answer = await self._invoke_pipeline(pipeline=pipeline, question=question, chat_history=chat_history)
        return answer
        
    @abstractmethod
    async def create_pipeline(self, config: RunnableConfig) -> StreamQAPipeline:
        ...

    async def _invoke_pipeline(self, pipeline: StreamQAPipeline, question: str, chat_history: Optional[Union[List[BaseMessage], List[Tuple[str, str]]]] = None):
        if not chat_history:
            chat_history = []
        question = question.strip().strip(",")
        outputs = None
        if self._stream_events:
            async for event in pipeline.stream(question=question, chat_history=chat_history, raise_if_error=True):
                if self._callback:
                    await self._callback(event)
        else:
            outputs = await pipeline.invoke(question=question, chat_history=chat_history, raise_if_error=True)
            if self._callback:
                await self._callback(outputs)
        outputs = pipeline.final_state
        logger.info(f"RAG outputs: {outputs}")
        return outputs.answer

class WebSearchTool(Tool):
    def __init__(self,
                 stream_events: bool,
                 agent_id: str,
                 model: str,
                 model_type: ModelType = ModelType.llama3,
                 fast_search: bool = True,
                 report_type: str = "research_report",
                 retrievers: str = "duckduckgo",
                 embeddings: Embeddings = None,
                 tokenizer: str = "/home/cunning/llama-3/Meta-Llama-3-8B",
                 max_input_tokens: int = 6144,
                 max_chat_history_token_length: int = 24576,
                 tool_name: Optional[str] = None,
                 tool_description: Optional[str] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 acronyms_filepath: Optional[str] = None,
                 langfuse_handler: Optional[CallbackHandler] = None,
                 callback: Optional[Callable[[Union[PipelineEvent,
                                                    WebIntermediateOutputs]], Awaitable[None]]] = None,
                 ) -> None:
        self._agent_id = agent_id
        self._model = model
        self._model_type = model_type
        self._model_kwargs = model_kwargs or {
            "temperature": 0,
            "top_p": 0.95,
            "max_tokens": 1024,
            "openai_api_key": "token-abc123",
            "openai_api_base": f"http://a.dgx:50080/qwen2-72b/v1"
        }
        self._report_type = report_type
        self._gpt_researcher = None
        self._max_input_tokens = max_input_tokens
        self._max_chat_history_token_length = max_chat_history_token_length
        self._tokenizer = tokenizer
        self._fast_search = fast_search
        self._retrievers = retrievers
        self._embeddings = embeddings
        self._acronyms_filepath = acronyms_filepath
        self._langfuse_handler = langfuse_handler
        self._callback = callback

        self._tool_name = tool_name or "web_search"
        self._description = tool_description or "Web-search instrument to extract information on the query from the Internet. May have some inaccurate information so you may need to specify the query based on the search results."

        super().__init__(stream_events)

    async def create_pipeline(self, config: RunnableConfig) -> str:
        web_pipe = WebSearchPipeline(
            agent_id=self._agent_id,
            model=self._model,
            model_type=self._model_type,
            fast_search=self._fast_search,
            report_type=self._report_type,
            retrievers=self._retrievers,
            embeddings=self._embeddings,
            tokenizer=self._tokenizer,
            max_input_tokens=self._max_input_tokens,
            max_chat_history_token_length=self._max_chat_history_token_length,
            model_kwargs=self._model_kwargs,
            acronyms_filepath=self._acronyms_filepath,
            langfuse_handler=self._langfuse_handler,
            runnable_config=config,
        )
        return web_pipe
    
class RagTool(Tool):
    def __init__(self,
                 stream_events: bool,
                 tool_name: Optional[str] = None,
                 tool_description: Optional[str] = None,
                 ) -> None:
        super().__init__(stream_events)
        raise NotImplementedError("RagTool is not implemented")

class ContextExtractionTool(Tool):
    def __init__(self,
                 stream_events: bool,
                 tool_name: Optional[str] = None,
                 tool_description: Optional[str] = None,
                 ) -> None:
        self._tool_name = tool_name or "context_extraction"
        self._description = tool_description or "Обращение с вопросом к специалисту по извлечению контекста из текста. Важно: у этого специалиста самый низкий приоритет, вопрос ему нужно переадресовывать, только если нет других подходящих, или если пользователь попросит об этом напрямую"
        super().__init__(stream_events)
    raise NotImplementedError("ContextExtractionTool is not implemented")