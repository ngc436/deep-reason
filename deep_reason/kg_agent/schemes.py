from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass

from deep_reason.schemes import Chunk


@dataclass
class ChunkTuple:
    """Represents a tuple of chunks with current chunk and optional context chunks"""
    current_chunk: Chunk
    left_context: Optional[Chunk] = None
    right_context: Optional[Chunk] = None

# Raw triplets
class Triplet(BaseModel):
    triplet_id: str = Field(description="Unique identifier for the triplet")
    chunk_id: str = Field(description="Unique identifier for the chunk where the triplet was found")
    subject: str = Field(description="The entity performing the action or having the property considering the chunk")
    predicate: str = Field(description="The relationship or action considering the chunk")
    object: str = Field(description="The entity receiving the action or the value of the property considering the chunk")

class TripletList(BaseModel):
    triplets: List[Triplet] = Field(description="List of knowledge triplets extracted from the text")

# Ontology nodes and relations
class OntologyNode(BaseModel):
    node_id: str = Field(description="Unique id of the ontology node")
    entity: str = Field(description="Entity name (Entity class name)")


class OntologyRelation(BaseModel):
    relation_id: int = Field(description="Unique id of the relation")
    relation_name: str = Field(description="Name of the relation (Relation class name)")


class OntologyNodesConnection(BaseModel):
    node_id_1: str = Field(description="Id of the first ontology node (class)")
    node_id_2: str = Field(description="Id of the second ontology node (class)")
    relation_id: int = Field(description="Id of the relation (Relation class)")


class OntologyStructure(BaseModel):
    nodes: List[OntologyNode] = Field(description="List of ontology nodes")
    relations: List[OntologyRelation] = Field(description="List of ontology relations")
    connections: List[OntologyNodesConnection] = Field(description="List of connections between ontology nodes")


# Knowledge graph nodes and relations
class KgNode(BaseModel):
    node_id: str = Field(description="Unique identifier of the knowledge graph node")
    entity_name: str = Field(description="Entity name")
    ontology_node_id: str = Field(description="Id of the ontology node this entity is instance of")


class KgTriplet(BaseModel):
    kg_subject_id: str = Field(description="Id of a KgNode that is a subject in this triplet")
    kg_object_id: str = Field(description="Id of a KgNode that is an object in this triplet")
    ontology_nodes_connection_id: str = Field(description="Id of the ontology nodes connection that describes predicate for this two entities")


class KgStructure(BaseModel):
    kg_nodes: List[KgNode] = Field(description="List of knowledge graph nodes")
    kg_triplets: List[KgTriplet] = Field(description="List of knowledge graph triplets combined from raw initial triplets")


class KGMiningWorkflowState(BaseModel):
    no_cache: bool = Field(default=False, description="If true, the workflow will not use cached results")  
    chunks: List[Chunk]
    triplets: Optional[List[Triplet]] = None
    ontology: Optional[OntologyStructure] = None
    knowledge_graph: Optional[KgStructure] = None


class AggregationInput(BaseModel):
    items: List[Triplet] = Field(..., description="List of triplets to process")
    input: Dict[str, Any] = Field(..., description="Additional input for the internal refine chain")


class KGMiningResult(BaseModel):
    """Result of knowledge graph mining containing triplets, ontology and knowledge graph."""
    triplets: List[Triplet] = Field(default_factory=list, description="The extracted triplets")
    ontology: Optional[OntologyStructure] = Field(default=None, description="The constructed ontology")
    knowledge_graph: Optional[KgStructure] = Field(default=None, description="The constructed knowledge graph")

