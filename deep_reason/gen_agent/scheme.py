from typing import Dict, List
from pydantic import BaseModel, Field

class ComplexRelationship(BaseModel):
    """Model for complex relationship between chain endpoints"""
    first_entity: str = Field(description="First entity in the chain")
    last_entity: str = Field(description="Last entity in the chain")
    evidence: List[str] = Field(description="List of evidence to infer relationship between first and last entities in the chain based on the relationships and descriptions.")
    relationships: List[str] = Field(description="Inferred relationships between first and last entities in a form of concise sentences. Each of the two entities should be in the sentence.")

class ComplexRelationshipResult(BaseModel):
    """Model for the complete result of complex relationship inference"""
    chain: List[str] = Field(description="The chain of entities")
    first_entity: str = Field(description="First entity in the chain")
    last_entity: str = Field(description="Last entity in the chain")
    entity_descriptions: Dict[str, str] = Field(description="Descriptions of entities in the chain")
    relationship_descriptions: List[str] = Field(description="Descriptions of relationships between consecutive entities")
    inferred_relationships: List[str] = Field(description="Inferred relationships between first and last entities")
    evidence: List[str] = Field(description="Evidence supporting the inferred relationships")

class KnowledgeEditingInput(BaseModel):
    """Model for the input data for knowledge editing"""
    edit_prompt: str = Field(description="Input prompt in a form of question where answer is one of the entities")
    subject: str = Field(description="The subject of the edit prompt should point to the target entity")
    target: str = Field(description="The actual answer to the edit prompt that is one of the entities")
    generalization_prompt: str = Field(description="The generalization prompt to check that editing is successful with a bit changed edit prompt")
    locality_prompt: str = Field(description="The locality prompt to check that model does not influenced on unrelated to editing inputs (though connected by entity)")
    portability_prompt: str = Field(description="The portability prompt to measure success for reasoning or application") 