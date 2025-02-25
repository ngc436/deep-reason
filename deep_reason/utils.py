from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, AddableValuesDict
from langchain_core.runnables.base import PipelineEvent
from transformers import PreTrainedTokenizer

Input = TypeVar("Input", covariant=True)
StateT = TypeVar("StateT", covariant=True)


class QAPipeline(Generic[StateT], ABC):
    @abstractmethod
    def build_chain(self) -> Runnable[StateT, AddableValuesDict]:
        ...

    @abstractmethod
    async def batch(self,
                    questions: List[str],
                    raise_if_error: bool = False,
                    show_progress: bool = True,
                    max_concurrency: Optional[int] = None) -> List[StateT]:
        ...

    @abstractmethod
    async def invoke(self,
                     question: str,
                     chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]] = None,
                     raise_if_error: bool = False) -> StateT:
        ...


class StreamQAPipeline(QAPipeline[StateT]):
    final_state: Optional[StateT]

    @staticmethod
    def _convert_chat_history(chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]]) -> List[BaseMessage]:
        def _convert_tuple(msg_type: str, msg: str) -> BaseMessage:
            supported_msg_types = ["system", "human", "ai"]

            match msg_type:
                case "system":
                    message = SystemMessage(content=msg)
                case "human":
                    message = HumanMessage(content=msg)
                case "ai":
                    message = AIMessage(content=msg)
                case _:
                    raise ValueError(f"Unsupported message type for ({msg_type}, {msg}). "
                                     f"Supported types: {supported_msg_types}")

            return message

        if not chat_history or len(chat_history) == 0 or isinstance(chat_history[0], BaseMessage):
            return chat_history
        elif isinstance(chat_history[0], tuple):
            return [_convert_tuple(msg_type, msg) for msg_type, msg in chat_history]
        else:
            raise ValueError("Unsupported types of list values for chat_history. "
                             "It should be either BaseMessage or Tuple")

    @staticmethod
    def _cut_chat_history(chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]],
                          threshold: int,
                          tokenizer: PreTrainedTokenizer) -> List[BaseMessage]:
        if not chat_history or len(chat_history) == 0:
            return chat_history
        res = []
        s = 0
        for message in reversed(chat_history):
            l = len(tokenizer.tokenize(message.content))
            if s + l > threshold:
                break
            s += l
            res.append(message)
        return res[::-1]

    @abstractmethod
    async def stream(self,
                     question: str,
                     chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]] = None,
                     raise_if_error: bool = False,
                     only_eos_answer_event: bool = False) -> AsyncIterator[PipelineEvent]:
        ...

    @abstractmethod
    async def events(self,
                     question: str,
                     chat_history: Optional[List[BaseMessage] | List[Tuple[str, str]]] = None,
                     raise_if_error: bool = False,
                     only_eos_answer_event: bool = False) -> List[PipelineEvent]:
        ...
