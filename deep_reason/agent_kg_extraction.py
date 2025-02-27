import os
import logging
from typing import Annotated, Dict, Any, List, Optional, Callable, Awaitable

import chromadb
from transformers import PreTrainedTokenizer
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langfuse.callback import CallbackHandler

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from deep_reason.utils import StreamQAPipeline
from deep_reason.schemes import AgentIntermediateOutputs
from deep_reason.tools import Tool

logger = logging.getLogger(__name__)

class KgAgent(StreamQAPipeline[AgentIntermediateOutputs]):
    def __init__(self,
                 *,
                 agent_id: str,
                 model: str,
                 model_kwargs: Dict[str, Any],
                 tools: List[Tool],
                 tokenizer: PreTrainedTokenizer,
                 file_saving_path: str,
                 max_input_tokens: int = 6144,
                 max_chat_history_token_length: int = 24576,
                 file_saving_callback: Optional[Callable[[Any], Awaitable[None]]] = None,
                 langfuse_handler: Optional[CallbackHandler] = None,
                 agent_prompt: Optional[str] = None):
        # self.chroma_client = chromadb.Client()
        # self.collection = self.chroma_client.get_or_create_collection("kg_collection")
        self._agent_id = agent_id
        self._model = model
        self._model_kwargs = model_kwargs
        self._tools = [
            *(item.tool for item in tools),
            StructuredTool.from_function(coroutine=self.save_document)
        ]
        self._tokenizer = tokenizer
        self._file_saving_path = file_saving_path
        self._max_input_tokens = max_input_tokens
        self._max_chat_history_token_length = max_chat_history_token_length
        self._file_saving_callback = file_saving_callback

        if not os.path.exists(self._file_saving_path):
            logger.info(f"path {self._file_saving_path} not found, creating")
            os.mkdir(self._file_saving_path)

        self._last_message = None

        chat_model = ChatOpenAI(
            name="llm",
            model="/model",
            streaming=False,
            **model_kwargs
        )
        self._agent = create_react_agent(
            chat_model, self._tools, messages_modifier=agent_prompt or SYS_PROMPT)
        self._config = {"callbacks": [
            langfuse_handler]} if langfuse_handler else {}

    async def save_document(self, document_name: str):
        """Сохранение содержимого текущего сообщения чата после маркера '<Содержимое файла>' и до маркера '<Конец содержимого>' в файл с указанным названием. Маркеры содержимого ДОЛЖНЫ присутствовать в том же сообщении что и вызов инструмента"""

        def _extract_content(message):
            try:
                content = message.split("<Содержимое файла>", maxsplit=1)[1]
                return content.split("<Конец содержимого>", maxsplit=1)[0]
            except IndexError:
                raise ContentExtractionError("Ошибка сохранения: некорректные маркеры содержимого")
        
        filepath = os.path.join(self._file_saving_path, document_name)
        callback_outputs = None
        try:
            content = _extract_content(self._last_message)
            with open(filepath, "w+") as f:
                f.write(content)
            callback_outputs = FileSavingOutputs(filepath=filepath)
        except Exception as e:
            callback_outputs = FileSavingOutputs(filepath=filepath, error=e)
            raise e
        finally:
            if self._file_saving_callback:
                await self._file_saving_callback(callback_outputs)
        return f"Документ сохранен под именем {document_name}"



    def add_entity(self, entity: dict):
        self.collection.add(entity)