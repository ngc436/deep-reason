from pydantic import BaseModel, Field
import uuid


def uid_factory() -> str:
    return str(uuid.uuid4())

class PipelineEvent(BaseModel):
    uid: str = Field(default_factory=uid_factory)
    agent_id: str = Field(default_factory=uid_factory)
    name: str

class Chunk(BaseModel):
    text: str | None = Field(description="Text of the chunk")
    chapter_name: str | None = Field(description="Name of the chapter")
    document_id: int = Field(description="Id of the document")
    order_id: int = Field(description="Order id of the chunk in the document")

class QAnswer(BaseModel):
    answer: str = Field(description="Answer to the question")