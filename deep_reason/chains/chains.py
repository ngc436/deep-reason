from contextlib import contextmanager
from datetime import datetime
import json
import logging
import pandas as pd
import os
from typing import Any, Dict, List, Optional, Tuple, cast, Union, TypeVar, Generic, Callable, Protocol
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import asyncio
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate
from langchain.output_parsers import RetryOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import LanguageModelInput
from dataclasses import dataclass
from itertools import groupby
from abc import ABC, abstractmethod

from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.schemes import Chunk
from deep_reason.utils import VLLMChatOpenAI
from examples.kg_extraction import load_obliqa_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import tiktoken


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)", 
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@contextmanager
def measure_time(name: Optional[str] = None):
    start_time = datetime.now()
    yield
    end_time = datetime.now()

    suffix = f"for {name}" if name else ""
    logger.info(f"Time taken {suffix}: {(end_time - start_time).total_seconds()} seconds")


def build_chain(llm: BaseChatModel, 
                system_template: str,
                human_template: str,
                parser: Optional[PydanticOutputParser] = None) -> Runnable:
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template([{"text": system_template}]),
            HumanMessagePromptTemplate.from_template([{"text": human_template}])
        ]
    )
    
    chain = (
        prompt
        | RunnableParallel(completion=llm, prompt_value=RunnablePassthrough())
    )

    if parser:
        retry_planner_parser = RetryOutputParser.from_llm(
            parser=parser,
            llm=llm,
            prompt=PromptTemplate.from_template("{prompt}"),
            max_retries=3
        )
        
        def _do_parsing_retrying(x: dict):
                result = None
                completion = x['completion'].content
                prompt_value = x['prompt_value']

                logger.info(f"Trying structured parsing, Received completion: {completion}")

                try:
                    result = retry_planner_parser.parse_with_prompt(completion=completion,prompt_value=prompt_value)
                except OutputParserException as e:
                    logger.warning("Proceeding without result due to parser errors (even after retrying). "
                                   "Prompt - %s" % prompt_value)
                    raise e                    

                # return {"completion": result, "prompt_value": prompt_value}
                return result

        chain = (
            RunnableLambda(lambda x: {**x, "response_format_description": parser.get_format_instructions()})
            | chain
            | RunnableLambda(_do_parsing_retrying, name="retry_planner_lambda")
        )
    # else:
    #     chain = chain | StrOutputParser()  

    return chain


@dataclass
class ChunkTuple:
    """Represents a tuple of chunks with current chunk and optional context chunks"""
    current_chunk: Chunk
    left_context: Optional[Chunk] = None
    right_context: Optional[Chunk] = None


def build_triplets_mining_chain(llm: BaseChatModel) -> Runnable:
    system_template = """
    # Instruction for Creating Nodes and Triplets on a text fragment
    You are an expert knowledge graph engineer. Extract knowledge triplets from the provided text chunk.
A knowledge triplet consists of (subject, relation, object) where:
- subject is the entity performing the action or having the property
- relation is the relationship or action
- object is the entity receiving the action or the value of the property

Consider the context around the current chunk to ensure coherent extraction.
Use a structured triplet format to capture data, as follows: "subject, relation, object". 
For example, from "Albert Einstein, born in Germany, is known for developing the theory of relativity," extract "Albert Einstein, country of birth, Germany; Albert Einstein, developed, Theory of Relativity." 
Remember that you should break complex triplets like "John, position, engineer in Google" into simple triplets like "John, position, engineer", "John, work at, Google". 
Length of your triplet should not be more than 7 words. 
You should extract only concrete knowledges, any assumptions must be described as hypothesis. 
For example, from phrase "John have scored many points and potentially will be winner" you should extract "John, scored many, points; John, could be, winner" and should not extract "John, will be, winner". 
Remember that object and subject must be an atomary units while relation can be more complex and long. 
If observation states that you take item, the triplet shoud be: 'item, is in, inventory' and nothing else.

Do not miss important information. 
If observation is 'book involves story about knight, who needs to kill a dragon', triplets should be 'book, involves, knight', 'knight, needs to kill, dragon'. 
If observation involves some type of notes, do not forget to include triplets about entities this note includes. 
There could be connections between distinct parts of observations. 
For example if there is information in the beginning of the observation that you are in location, and in the end it states that there is an exit to the east, you should extract triplet: 'location, has exit, east'. Several triplets can be extracted, that contain information about the same node. For example 'kitchen, contains, apple', 'kitchen, contains, table', 'apple, is on, table'.

Do not miss this type of connections. 
Other examples of triplets: 'room z, contains, black locker'; 'room x, has exit, east', 'apple, is on, table', 'key, is in, locker', 'apple, to be, grilled', 'potato, to be, sliced', 'stove, used for, frying', 'recipe, requires, green apple', 'recipe, requires, potato'. Do not include triplets that state the current location of an agent like 'you, are in, location'. 
Do not use 'none' as one of the entities. 
If there is information that you read something, do not forget to incluse triplets that state that entity that you read contains information that you extract.

Remember that triplets must be extracted in format:
{response_format_description}"""

    human_template = """Extract knowledge triplets from the following text chunk:
    
Current chunk: {current_chunk}

{left_context_prefix}{left_context}

{right_context_prefix}{right_context}

{response_format_description}"""
    
    parser = PydanticOutputParser(pydantic_object=TripletList)
    
    return build_chain(llm, system_template, human_template, parser)

# def build_kg_construction_chain(llm: BaseChatModel) -> Runnable['RefinerInput', Dict[str, Any]]:
#     system_template = """You are an expert knowledge graph engineer. Your task is to organize knowledge triplets 
#     by deduplicating entities and relationships that refer to the same concept.

#     For each triplet (subject, predicate, object):
#     1. Identify when different subjects or objects refer to the same entity
#     2. Identify when different predicates express the same relationship
#     3. Create a normalized representation by combining similar entities and relationships
    
#     The goal is to create a more cohesive and less redundant set of triplets where:
#     - Multiple variations of the same entity name are consolidated (e.g., "John Smith", "J. Smith", "Mr. Smith" → "John Smith")
#     - Equivalent relationships are standardized (e.g., "works for", "is employed by", "works at" → "works for")
    
#     Your response should be in the following format:
#     {response_format_description}"""

#     human_template = """Process the following knowledge triplets to deduplicate entities and relationships:

#     Triplets:
#     {triplet_list}

#     Identify similar entities and relationships, and provide a normalized representation.
#     {response_format_description}"""
    
#     parser = PydanticOutputParser(pydantic_object=KgTripletList)
    
#     chain = build_chain(llm, system_template, human_template, parser)
    
#     def _process_input(refiner_input: RefinerInput):
#         # Extract triplets from items list
#         formatted_triplets = "\n".join([
#             f"- Subject: {item.subject}, Predicate: {item.predicate}, Object: {item.object}"
#             for item in refiner_input.items
#         ])
        
#         return {
#             "triplet_list": formatted_triplets
#         }
    
#     def _process_output(result: KgTripletList) -> Dict[str, Any]:
#         # Return the deduplicated triplets
#         return {
#             "kg_triplet_list": result.kg_triplets
#         }
    
#     return RunnableLambda(_process_input) | chain | RunnableLambda(_process_output)


def build_ontology_refinement_chain(llm: BaseChatModel) -> Runnable['RefinerInput', Dict[str, Any]]:
    system_template = """You are an expert knowledge graph engineer. Your task is to construct and refine an ontology 
    based on knowledge triplets extracted from text. An ontology consists of entity types and relationship types.

    For each triplet (subject, predicate, object):
    1. Identify entity types for both subject and object
    2. Categorize relationship types for predicates
    3. Organize these into a hierarchical structure where appropriate
    
    Ontology nodes should depict entities or concepts, similar to Wikipedia high-level nodes. 
    Look carefully at the provided triplets and existing so far ontology and decide the following:
    1. If the provided triplet can be added to the existing ontology, add it to the ontology.
    2. If the provided triplet does not fit into the existing ontology, create a new ontology node for it.
    
    Your response should be in the following format:
    {response_format_description}
    """

    human_template = """Process the following knowledge triplets to create or refine an ontology:

    Triplets:
    {kg_triplet_list}

    Current Ontology:
    {current_ontology}

    Create or refine the ontology to categorize entity types and relationship types.
    {response_format_description}"""
    
    parser = PydanticOutputParser(pydantic_object=OntologyStructure)
    
    chain = build_chain(llm, system_template, human_template, parser)
    
    def _process_input(refiner_input: RefinerInput):
        # Extract triplets from items list
        current_ontology = refiner_input.input.get("current_ontology", None)
        if not current_ontology:
            logger.info("No current ontology provided, creating empty ontology")
            current_ontology = OntologyStructure(nodes=[], relations=[], connections=[])
        
        formatted_triplets = "\n".join([
            f"- Subject: {item.subject}, Predicate: {item.predicate}, Object: {item.object}"
            for item in refiner_input.items
        ])
        
        return {
            "triplet_list": formatted_triplets,
            "current_ontology": current_ontology
        }
    
    def _process_output(result: OntologyStructure) -> Dict[str, Any]:
        # Extract the ontology dictionary to be used in the next iteration
        return { "current_ontology": result }
    
    return RunnableLambda(_process_input) | chain | RunnableLambda(_process_output)


def build_kg_refining_chain(llm: BaseChatModel) -> Runnable['RefinerInput', Dict[str, Any]]:
    system_template = """You are an expert knowledge graph engineer. Your task is to build a knowledge graph using the provided triplets and ontology.

    The knowledge graph consists of a set of triplets that contain:
    1. Entities - unique nodes in the graph with types from the ontology
    2. Relationships - connections between entities based on the triplets
    
    Use the ontology to categorize entities and relationships appropriately. Do not modify the ontology.
    Reduce the amount of relationships in current knowledge graph if possible and update the corresponding triplet_ids field.
    
    Compile or refine the knowledge graph based on these triplets and the existing ontology.
    {response_format_description}"""

    human_template = """Process the following knowledge triplets to build or refine a knowledge graph:

    Triplets:
    {triplet_list}

    Current Ontology (DO NOT MODIFY):
    {ontology}

    Current Knowledge Graph:
    {current_graph}
    """
    
    parser = PydanticOutputParser(pydantic_object=KgStructure)
    
    chain = build_chain(llm, system_template, human_template, parser)
    
    def _process_input(refiner_input: RefinerInput):
        ontology = refiner_input.input.get("ontology", None)
        if not ontology:
            raise KGConstructionAgentException("No ontology provided for kg refining chain")
        
        current_graph = refiner_input.input.get("current_graph", None)
        if not current_graph:
            logger.info("No current graph provided, creating empty graph")
            current_graph = KgStructure(kg_triplets=[])

        # Extract triplets from items list
        formatted_triplets = "\n".join([
            f"- Subject: {item.subject}, Predicate: {item.predicate}, Object: {item.object}"
            for item in refiner_input.items
        ])
        
        return {
            "triplet_list": formatted_triplets,
            "ontology": ontology,
            "current_graph": current_graph
        }
    
    def _process_output(x: Dict[str, Any]) -> Dict[str, Any]:
        # Return a dictionary with entities and relationships for the next iteration
        result = x["result"]
        ontology = x["inputs"]["ontology"]
        return {
            "ontology": ontology,
            "current_graph": result
        }
    
    return (
        RunnableLambda(_process_input) 
        | RunnableParallel(result=chain, inputs=RunnablePassthrough()) 
        | RunnableLambda(_process_output)
    )

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
    kg_triplets: List[KgTriplet] = Field(description="List of knowledge graph triplets combined from raw initial triplets")


class KGMiningWorkflowState(BaseModel):
    chunks: List[Chunk]
    triplets: Optional[List[Triplet]] = Field(None, default_factory=list)
    ontology: Optional[OntologyStructure] = Field(None, default_factory=dict)
    knowledge_graph: Optional[KgStructure] = Field(None, default_factory=list)


class KGConstructionAgentException(Exception):
    pass


class RefinerInput(BaseModel):
    items: List[Triplet] = Field(..., description="List of triplets to process")
    input: Dict[str, Any] = Field(..., description="Additional input for the internal refine chain")


class Refiner(Runnable[RefinerInput, Dict[str, Any]]):
    """Abstract class for refining data in batches with LLM chains"""
    
    def __init__(self, refine_chain: Runnable[RefinerInput, Dict[str, Any]], tokenizer=None, context_window_length: int = 8000):
        self.refine_chain = refine_chain
        self.context_window_length = context_window_length
        if tokenizer is None:
            # default tokenizer if nothing provided
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
    
    def _estimate_size_in_tokens(self, state: Dict[str, Any] | BaseModel | str) -> str:
        if isinstance(state, BaseModel):
            text = state.model_dump_json()
        elif isinstance(state, str):
            text = state
        else:
            text = json.dumps(state)
    
        return len(self.tokenizer.encode(text))

    def _create_batch(self, remaining_items: List[Triplet], current_state: Dict[str, Any] | BaseModel | str) -> Tuple[RefinerInput, List[Triplet]]:
        """Create a batch of items that fits within the context window"""
        # Calculate current state size
        state_size = self._estimate_size_in_tokens(current_state)       
        # Dynamically form the next batch based on current state size
        current_batch = []
        current_batch_size = 0
        max_batch_size = (self.context_window_length - state_size) // 2
        
        # Make a copy of remaining items to avoid modifying the original during batch creation
        updated_remaining = remaining_items.copy()
        
        # Add items to the batch until we reach the size limit
        while updated_remaining and current_batch_size < max_batch_size:
            item = updated_remaining[0]
            item_size = self._estimate_size_in_tokens(item)
            
            if current_batch_size + item_size <= max_batch_size:
                current_batch.append(item)
                current_batch_size += item_size
                updated_remaining.pop(0)
            else:
                # This item would exceed the batch size limit
                break
        
        # If we couldn't fit any items, take at least one (necessary for progress)
        if not current_batch and updated_remaining:
            raise KGConstructionAgentException(
                "Cannot pack a single item into the batch due to too long context (most probably due to too long result generated on the previous iteration)"
            )
            
        return RefinerInput(items=current_batch, input=current_state), updated_remaining
    
    async def ainvoke(self, refiner_input: RefinerInput, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process items in batches based on context window limitations"""
        current_state = refiner_input.input
        remaining_items = refiner_input.items.copy()
        batch_idx = 0
        batch_name = self.__class__.__name__
        
        while remaining_items:
            try:
                # Create the next batch based on context window limitations
                current_batch, remaining_items = self._create_batch(remaining_items, current_state)
                
                batch_idx += 1
                logger.info(f"Processing {batch_name} batch {batch_idx} with {len(current_batch)} items. Remaining: {len(remaining_items)}")
                
                # Process batch
                with measure_time(f"processing {batch_name} batch {batch_idx}"):
                    current_state = await self.refine_chain.ainvoke(current_batch, config=config)
                
            except Exception as e:
                # Handle exceptions
                error_msg = f"Error processing {batch_name} batch {batch_idx}: {str(e)}"
                logger.error(error_msg)
                raise KGConstructionAgentException(error_msg)
        
        return current_state
    
    def invoke(self, refiner_input: RefinerInput, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version that runs the async method in an event loop"""
        return asyncio.run(self.ainvoke(refiner_input, config))


class KGConstructionAgent:
    def __init__(self, llm: BaseChatModel, tokenizer: Optional[PreTrainedTokenizerBase | str] = None, context_window_length: int = 8000):
        self.llm = llm
        self.context_window_length = context_window_length
        if tokenizer is None:
            # default tokenizer if nothing provided
            # better than nothing
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer


    async def triplets_mining(self, state: KGMiningWorkflowState, config: Optional[Dict[str, Any]] = None) -> KGMiningWorkflowState:
        # 1. Order chunks by document_id and order_id
        chunks = sorted(state.chunks, key=lambda x: (x.document_id, x.order_id))
        
        # 2. Create chunk tuples with sliding window approach
        chunk_tuples = []
        
        # Group chunks by document_id
        for doc_id, doc_chunks in groupby(chunks, key=lambda x: x.document_id):
            doc_chunks_list = list(doc_chunks)
            
            for i, chunk in enumerate(doc_chunks_list):
                left_context = None if i == 0 else doc_chunks_list[i-1]
                right_context = None if i == len(doc_chunks_list) - 1 else doc_chunks_list[i+1]
                
                chunk_tuples.append(ChunkTuple(
                    current_chunk=chunk,
                    left_context=left_context,
                    right_context=right_context
                ))
        
        # 3. Build the triplet mining chain
        chain = build_triplets_mining_chain(self.llm)
        
        # 4. Process chunk tuples using batch method
        inputs = []
        for ct in chunk_tuples:
            input_dict = {
                "current_chunk": ct.current_chunk.text,
                "left_context_prefix": "Left context: " if ct.left_context else "No left context available.",
                "left_context": ct.left_context.text if ct.left_context else "",
                "right_context_prefix": "Right context: " if ct.right_context else "No right context available.",
                "right_context": ct.right_context.text if ct.right_context else ""
            }
            inputs.append(input_dict)
        
        results = await chain.abatch(inputs, return_exceptions=True, config=config)
        
        # 5. Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {chunk_tuples[i].current_chunk.chapter_name}: {result}")
            else:
                # Attach the chunk information to each result
                for triplet in result.triplets:
                    valid_results.append({
                        "chunk": chunk_tuples[i].current_chunk,
                        "triplet": triplet
                    })
        
        # 6. Sort triplets by document and then chunk order
        valid_results = sorted(
            valid_results, 
            key=lambda x: (x["chunk"].document_id, x["chunk"].order_id)
        )

        # 7. Create and return new state with valid results
        return KGMiningWorkflowState(
            chunks=state.chunks,
            triplets=valid_results,
            ontology=state.ontology,
            knowledge_graph=state.knowledge_graph
        )
    
    async def ontology_refining(self, state: KGMiningWorkflowState, config: Optional[Dict[str, Any]] = None) -> KGMiningWorkflowState:
        # Check for empty triplets
        if not state.triplets:
            logger.error("No triplets found in state, cannot construct ontology")
            raise KGConstructionAgentException("Cannot construct ontology from empty triplets")
        
        refiner = Refiner(
            refine_chain=build_ontology_refinement_chain(self.llm), 
            tokenizer=self.tokenizer, 
            context_window_length=self.context_window_length
        )
        refiner_input = RefinerInput(items=state.triplets, input={"current_ontology": None})
        current_ontology = await refiner.ainvoke(input=refiner_input, config=config)
        
        # Return updated state with the refined ontology
        return KGMiningWorkflowState(
            chunks=state.chunks,
            triplets=state.triplets,
            ontology=current_ontology,
            knowledge_graph=state.knowledge_graph
        )

    async def kg_refining(self, state: KGMiningWorkflowState, config: Optional[Dict[str, Any]] = None) -> KGMiningWorkflowState:
        # Check for empty triplets or ontology
        if not state.triplets:
            logger.error("No triplets found in state, cannot build knowledge graph")
            raise KGConstructionAgentException("Cannot build knowledge graph without triplets")
        
        if not state.ontology:
            logger.error("No ontology found in state, cannot build knowledge graph")
            raise KGConstructionAgentException("Cannot build knowledge graph without ontology")
        
        refiner = Refiner(
            refine_chain=build_kg_refining_chain(self.llm), 
            tokenizer=self.tokenizer, 
            context_window_length=self.context_window_length
        )
        refiner_input = RefinerInput(
            items=state.triplets,
            input={
                "ontology": state.ontology,
                "current_graph": None
            }
        )
        current_kg = await refiner.ainvoke(input=refiner_input, config=config)
        
        # Return updated state with the refined knowledge graph
        return KGMiningWorkflowState(
            chunks=state.chunks,
            triplets=state.triplets,
            ontology=state.ontology,
            knowledge_graph=current_kg
        )

    def build_wf(self) -> Runnable[KGMiningWorkflowState, Dict[str, Any]]:
        wf = StateGraph(KGMiningWorkflowState)
        
        wf.add_node("triplets_mining", self.triplets_mining)
        wf.add_node("ontology_refining", self.ontology_refining)
        wf.add_node("kg_refining", self.kg_refining)

        wf.add_edge(START, "triplets_mining")
        wf.add_edge("triplets_mining", "ontology_refining")
        wf.add_edge("ontology_refining", "kg_refining")
        wf.add_edge("kg_refining", END)

        return wf.compile()


class KGMiningResult(BaseModel):
    """Result of knowledge graph mining containing triplets, ontology and knowledge graph."""
    triplets: List[Triplet] = Field(default_factory=list, description="The extracted triplets")
    ontology: Optional[OntologyStructure] = Field(default=None, description="The constructed ontology")
    knowledge_graph: Optional[KgStructure] = Field(default=None, description="The constructed knowledge graph")


async def run_kg_mining(llm: BaseChatModel, chunks: List[Chunk], output_path: str = "kg_output") -> KGMiningWorkflowState:
    agent = KGConstructionAgent(llm)
    state = KGMiningWorkflowState(chunks=chunks)
    result = await agent.build_wf().ainvoke(state)
    result = KGMiningWorkflowState.model_validate(result)
    
    # Create KGMiningResult object
    kg_mining_result = KGMiningResult(
        triplets=result.triplets if result.triplets else [],
        ontology=result.ontology,
        knowledge_graph=result.knowledge_graph
    )
    
    # Save results to file on disk
    os.makedirs(output_path, exist_ok=True)
    
    # Save complete result
    result_path = os.path.join(output_path, "kg_mining_result.json")
    with open(result_path, "w") as f:
        f.write(kg_mining_result.model_dump_json(indent=2))
    logger.info(f"Saved KG mining result to {result_path}")
    
    return result


def load_obliqa_dataset(obliqa_dir: str) -> List[Chunk]:
    all_chunks = []
    for fname in os.listdir(obliqa_dir):
        df = pd.read_json(f"{obliqa_dir}/{fname}", orient="records")
        for ix, row in df.iterrows():
            all_chunks.append(Chunk(text=row["Passage"], 
                                    chapter_name=str(row["PassageID"]), 
                                    document_id=row["DocumentID"], 
                                    order_id=ix))
    return all_chunks


def main():
    llm = VLLMChatOpenAI(
        model="/model",
        base_url=os.environ[OPENAI_API_BASE],
        api_key=os.environ[OPENAI_API_KEY],
        temperature=0.3,
        max_tokens=2048
    )

    chunks = load_obliqa_dataset(obliqa_dir="datasets/ObliQA/StructuredRegulatoryDocuments")

    asyncio.run(run_kg_mining(llm, chunks))

if __name__ == "__main__":
    main()
      