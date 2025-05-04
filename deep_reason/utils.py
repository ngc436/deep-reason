import os 
import pandas as pd
from abc import ABC, abstractmethod
from typing import (Generic, List, Optional, Tuple, TypeVar, Any, AsyncIterator, 
                    Union, Sequence)
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from transformers import PreTrainedTokenizer
from enum import Enum
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.pregel.io import AddableValuesDict
from deep_reason.schemes import PipelineEvent
from deep_reason.schemes import Chunk

MessageLikeRepresentation = Union[
    BaseMessage, list[str], tuple[str, str], str, dict[str, Any]
]

LanguageModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation]]
LanguageModelOutput = Union[BaseMessage, str]
LanguageModelLike = Runnable[LanguageModelInput, LanguageModelOutput]
LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", BaseMessage, str)

Input = TypeVar("Input", covariant=True)
StateT = TypeVar("StateT", covariant=True)

class ModelType(Enum):
    qwen2 = "qwen2"
    qwen2_reason = "qwen2_reason"
    gpt4o = "gpt4o"

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


class VLLMChatOpenAI(ChatOpenAI):
    _no_think: bool = False

    def __init__(self, *args, no_think: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__._no_think = no_think

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
        
        # Add /no_think system prompt if no_think is True
        if self.__class__._no_think:
            if "messages" in payload:
                # Insert /no_think as the first system message
                payload["messages"].insert(0, {"role": "system", "content": "/no_think"})
            else:
                # If no messages exist, create a new list with the /no_think system message
                payload["messages"] = [{"role": "system", "content": "/no_think"}]
        
        return payload

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


class PipelineException(Exception):
    pass


class PipelineMaxInputTokensExceededException(PipelineException):
    pass


class AnsweringEvent(PipelineEvent):
    output: Optional[str]
    delta: str
    name: str = 'answering'

class ErrorEvent(PipelineEvent):
    error: str
    is_eos: bool
    name: str = 'error'

class SpecialTokens:
    _token_map = {
        ModelType.qwen2: {
            "system_header": "<|im_start|>system",
            "user_header": "<|im_start|>user",
            "assistant_header": "<|im_start|>assistant",
            "eot": "<|im_end|>"
        }
    }

    def __init__(self, model_type: ModelType):
        if model_type not in self._token_map:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {list(ModelType)}")
        
        tokens = self._token_map[model_type]
        self.system_header = tokens["system_header"]
        self.user_header = tokens["user_header"]
        self.assistant_header = tokens["assistant_header"]
        self.eot = tokens["eot"]

def _get_chunks(text: str, chunk_size: int) -> List[str]:
    ''' Split text into chunks of equal size'''
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def load_obliqa_dataset(obliqa_dir: str, file_idx: None | List[int] = None) -> List[Chunk]:
    if file_idx is None:
        fnames = os.listdir(obliqa_dir)
    else:
        fnames = [f"{i}.json" for i in file_idx]
    all_chunks = []
    for fname in fnames:
        df = pd.read_json(f"{obliqa_dir}/{fname}", orient="records")
        for ix, row in df.iterrows():
            all_chunks.append(Chunk(text=row["Passage"], 
                                    chapter_name=str(row["PassageID"]), 
                                    document_id=row["DocumentID"], 
                                    order_id=ix))
    return all_chunks

def load_books_mx_dataset(books_mx_path: str) -> List[Chunk]:
    df = pd.read_json(books_mx_path, orient="records")
    all_chunks = []
    previous_document_fname = None
    chunk_ix = 0
    for ix, row in df.iterrows():
        current_document_fname =str(row["_source"]["metadata"]["file_name"])
        if current_document_fname != previous_document_fname:
            chunk_ix = 0
            previous_document_fname = current_document_fname
        all_chunks.append(Chunk(text=row["_source"]["paragraph"], 
                                chapter_name=str(row["_source"]["metadata"]["chapter"]), 
                                document_id=row["_source"]["metadata"]["idx"], 
                                order_id=chunk_ix))
        chunk_ix += 1
    return all_chunks


def parse_basic_auth(basic_auth_string: str) -> tuple[str, str]:
    """
    Parse a basic auth string in the format 'username:password' into a tuple of (username, password).
    
    Args:
        basic_auth_string: A string in the format 'username:password'
        
    Returns:
        A tuple of (username, password)
        
    Raises:
        ValueError: If the string is not in the correct format
    """
    basic_auth = basic_auth_string.split(":")
    if len(basic_auth) != 2:
        raise ValueError("Basic auth must be in the format 'username:password'")
    return (basic_auth[0], basic_auth[1])

