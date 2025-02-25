from pydantic import BaseModel, Field

class Triplet(BaseModel):
    subject: str = Field(description="Subject of the triplet")
    relation: str = Field(description="Relation between objects")
    object: str = Field(description="Object of the triplet")

class ChunkTripletsResult(BaseModel):
    triplets: list[Triplet] = Field(description="List of triplets")


