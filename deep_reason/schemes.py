from pydantic import BaseModel, Field
from typing import Annotated, Optional, List, Any, Dict
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.documents import Document
import uuid


def uid_factory() -> str:
    return str(uuid.uuid4())

class PipelineEvent(BaseModel):
    uid: str = Field(default_factory=uid_factory)
    agent_id: str = Field(default_factory=uid_factory)
    name: str

class Triplet(BaseModel):
    subject: str = Field(description="Subject of the triplet")
    relation: str = Field(description="Relation between objects")
    object: str = Field(description="Object of the triplet")


class Quadriplet(BaseModel):
    subject: str = Field(description="Subject of the quadriplet")
    relation: str = Field(description="Relation between subject and object")
    object: str = Field(description="Object of the quadriplet")
    timestamp: str = Field(description="Timestamp of the quadriplet")
    

class ChunkTripletsResult(BaseModel):
    triplets: list[Triplet] = Field(description="List of triplets")


class Chunk(BaseModel):
    text: str | None = Field(description="Text of the chunk")
    chapter_name: str | None = Field(description="Name of the chapter")
    document_id: int = Field(description="Id of the document")
    order_id: int = Field(description="Order id of the chunk in the document")

def _take_any(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a or b

class WebArbitraryOutputs(BaseModel):
    question: str
    answer: Optional[str] = None
    contexts: Optional[List[Document]] = None
    error: Optional[Exception] = None
    outputs: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

class WebIntermediateOutputs(BaseModel):
    # general
    agent_id: Annotated[str, _take_any]
    question: Annotated[str, _take_any]
    chat_history: Annotated[Optional[List[BaseMessage]], _take_any] = None

    # prepration: question contextualization, context expansion
    contextualized_question: Annotated[Optional[str], _take_any] = None
    acronyms_expansion: Annotated[Optional[str], _take_any] = None

    # web-search part
    #TODO: check types
    context_documents: Annotated[Optional[List[str]], _take_any] = None

    # answer generating part
    answer: Annotated[Optional[str], _take_any] = None

    # general if error occurs anywhere
    error: Annotated[Optional[Exception], _take_any] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def contexts(self) -> Optional[List[Document]]:
        return self.reranked_documents or self.retrieved_documents
    

class AgentIntermediateOutputs(BaseModel):
    # general
    agent_id: Annotated[str, _take_any]
    question: Annotated[str, _take_any]
    chat_history: Annotated[Optional[List[BaseMessage]], _take_any] = None

    #Tools results 
    tool_results: Annotated[Optional[List[ToolMessage]], _take_any] = None
    
    # answer generating part
    all_answers: Annotated[Optional[List[AIMessage]], _take_any] = None
    all_messages: Annotated[Optional[List[BaseMessage]], _take_any] = None
    answer: Annotated[Optional[str], _take_any] = None

    # general if error occurs anywhere
    error: Annotated[Optional[Exception], _take_any] = None

    class Config:
        arbitrary_types_allowed = True

class FileSavingOutputs(BaseModel):
    filepath: str
    error: Optional[Exception] = None

    class Config:
        arbitrary_types_allowed = True

