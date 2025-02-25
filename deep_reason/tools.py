import logging
from langchain.tools import BaseTool, StructuredTool
from langchain_core.runnables.base import Runnable
from langchain_core.runnables import RunnableConfig
from abc import abstractmethod
from typing import Optional, Union, List, Tuple
from deep_reason.utils import Input, StateT
from deep_reason.pipelines import StreamQAPipeline

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