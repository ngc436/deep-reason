from typing import Dict, List
from pydantic import BaseModel, Field

class ComplexRelationship(BaseModel):
    """Model for complex relationship between chain endpoints"""
    first_entity: str = Field(description="First entity in the chain")
    last_entity: str = Field(description="Last entity in the chain")
    evidence: List[str] = Field(description="List of evidence to infer relationship between first and last entities in the chain based on the relationships and descriptions.")
    relationships: List[str] = Field(description="Inferred relationships between first and last entities in a form of concise sentences. Each of the two entities should be in the sentence.")
    score: float = Field(description="Self-reflect on the quality of the inferred relationship, and give a score between 0 and 10. If relationship does not bring specific new information and very obvious or contain book-specific relationship like image or formula number, score should be close to 0. If relationship brings new information and complex, score should be close to 10.")
    reasoning: str = Field(description="Reasoning on the quality of the inferred relationship and score explanation")

class InferredRelationship(BaseModel):
    """Model for inferred relationship between two entities"""
    relationship: str = Field(description="Inferred relationship between two entities in a form of concise sentence. Each of the two entities should be in the sentence.")
    evidence: List[str] = Field(description="Evidence supporting the inferred relationship")
    score: float = Field(description="Self-reflect on the quality of the inferred relationship, and give a score between 0 and 10. If relationship does not bring specific new information and very obvious, score should be close to 0. If relationship brings new information and complex, score should be close to 10.")
    reasoning: str = Field(description="Reasoning on the quality of the inferred relationship and score explanation")

class ComplexRelationshipResult(BaseModel):
    """Model for the complete result of complex relationship inference"""
    chain: List[str] = Field(description="The chain of entities")
    first_entity: str = Field(description="First entity in the chain")
    last_entity: str = Field(description="Last entity in the chain")
    entity_descriptions: Dict[str, str] = Field(description="Descriptions of entities in the chain")
    relationship_descriptions: List[str] = Field(description="Descriptions of relationships between consecutive entities")
    inferred_relationships: List[InferredRelationship] = Field(description="Inferred relationships between first and last entities")

class Locality(BaseModel):
    """Model for locality of the edit prompt"""
    locality_prompt: str = Field(description="The locality prompt to check that model does not influenced on unrelated to editing inputs (though connected by entity)")
    locality_answer: str = Field(description="The answer to the locality prompt")

class Generalization(BaseModel):
    """Model for generalization of the edit prompt"""
    generalization_prompt: str = Field(description="The generalization prompt to check that editing is successful with a bit changed edit prompt")
    generalization_answer: str = Field(description="The answer to the generalization prompt")

class Portability(BaseModel):
    """Model for portability of the edit prompt"""
    portability_prompt: str = Field(description="The portability prompt to measure success for reasoning or application")
    portability_answer: str = Field(description="The answer to the portability prompt")

class KnowledgeEditingInput(BaseModel):
    """Model for the input data for knowledge editing"""
    edit_prompt: str = Field(description="Input prompt in a form of question where answer is one of the entities")
    subject: str = Field(description="The subject of the edit prompt should point to the target entity")
    target: str = Field(description="The actual answer to the edit prompt that is one of the entities")
    generalization: Generalization = Field(description="The generalization of the edit prompt")
    locality: Locality = Field(description="The locality of the edit prompt")
    portability: Portability = Field(description="The portability of the edit prompt")
    rephrase: List[str] = Field(description="Alternative ways to phrase the edit prompt")