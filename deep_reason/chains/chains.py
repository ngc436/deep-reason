from contextlib import contextmanager
from datetime import datetime
import json
import logging
import pandas as pd
import os
from typing import Any, Dict, List, Optional, cast, Union, TypeVar, Generic, Callable, Protocol
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
    system_template = """You are an expert knowledge graph engineer. Extract knowledge triplets from the provided text chunk.
A knowledge triplet consists of (subject, predicate, object) where:
- subject is the entity performing the action or having the property
- predicate is the relationship or action
- object is the entity receiving the action or the value of the property

Consider the context around the current chunk to ensure coherent extraction."""

    human_template = """Extract knowledge triplets from the following text chunk:
    
Current chunk: {current_chunk}

{left_context_prefix}{left_context}

{right_context_prefix}{right_context}

{response_format_description}"""
    
    parser = PydanticOutputParser(pydantic_object=TripletList)
    
    return build_chain(llm, system_template, human_template, parser)


def build_ontology_refinement_chain(llm: BaseChatModel) -> Runnable:
    system_template = """You are an expert knowledge graph engineer. Your task is to construct and refine an ontology 
    based on knowledge triplets extracted from text. An ontology consists of entity types and relationship types.

    For each triplet (subject, predicate, object):
    1. Identify entity types for both subject and object
    2. Categorize relationship types for predicates
    3. Organize these into a hierarchical structure where appropriate"""

    human_template = """Process the following knowledge triplets to create or refine an ontology:

    {triplets}

    Current Ontology:
    {current_ontology}

    Create or refine the ontology to categorize entity types and relationship types.
    {response_format_description}"""

    class OntologyStructure(BaseModel):
        ontology: Dict[str, List[str]] = Field(description="Categories of entities and relationships")
    
    parser = PydanticOutputParser(pydantic_object=OntologyStructure)
    
    return build_chain(llm, system_template, human_template, parser)


def build_kg_refining_chain(llm: BaseChatModel) -> Runnable:
    system_template = """You are an expert knowledge graph engineer. Your task is to build a knowledge graph using the provided triplets and ontology.

    The knowledge graph consists of:
    1. Entities - unique nodes in the graph with types from the ontology
    2. Relationships - connections between entities based on the triplets
    
    Use the ontology to categorize entities and relationships appropriately. Do not modify the ontology."""

    human_template = """Process the following knowledge triplets to build or refine a knowledge graph:

    {triplets}

    Current Ontology (DO NOT MODIFY):
    {current_ontology}

    Current Knowledge Graph:
    {current_kg}

    Compile or refine the knowledge graph based on these triplets and the existing ontology.
    {response_format_description}"""
    
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
    
    return build_chain(llm, system_template, human_template, parser)


class Triplet(BaseModel):
    subject: str = Field(description="The entity performing the action or having the property")
    predicate: str = Field(description="The relationship or action")
    object: str = Field(description="The entity receiving the action or the value of the property")


class TripletList(BaseModel):
    triplets: List[Triplet] = Field(description="List of knowledge triplets extracted from the text")


class Entity(BaseModel):
    id: str = Field(description="Unique identifier for the entity")
    name: str = Field(description="Entity name or label")
    type: str = Field(description="Entity type from the ontology")
    properties: Dict[str, str] = Field(description="Additional properties for the entity", default_factory=dict)


class Relationship(BaseModel):
    id: str = Field(description="Unique identifier for the relationship")
    source: str = Field(description="Source entity ID")
    target: str = Field(description="Target entity ID")
    type: str = Field(description="Relationship type from the ontology")
    properties: Dict[str, str] = Field(description="Additional properties for the relationship", default_factory=dict)


class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(description="Entities in the knowledge graph")
    relationships: List[Relationship] = Field(description="Relationships between entities")


class KGMiningWorkflowState(BaseModel):
    chunks: List[Chunk]
    triplets: List[Dict[str, Any]] = Field(default_factory=list)
    ontology: Dict[str, List[str]] = Field(default_factory=dict)
    knowledge_graph: List[Dict[str, Any]] = Field(default_factory=list)


class KGConstructionAgentException(Exception):
    pass


# Define a type variable for our batch input
T = TypeVar('T')
R = TypeVar('R')

class Refiner(Runnable[List[T], Dict[str, Any]], ABC):
    """Abstract class for refining data in batches with LLM chains"""
    
    def __init__(self, llm: BaseChatModel, tokenizer=None, context_window_length: int = 8000):
        self.llm = llm
        self.context_window_length = context_window_length
        if tokenizer is None:
            # default tokenizer if nothing provided
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        self.chain = self._build_chain()
    
    @abstractmethod
    def _build_chain(self) -> Runnable:
        """Build the LLM chain used for refining"""
        pass
    
    @abstractmethod
    def _prepare_batch_input(self, batch: List[T], current_result: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for the chain based on current batch and results"""
        pass
    
    @abstractmethod
    def _create_batch(self, remaining_items: List[T], current_result: Dict[str, Any]) -> tuple[List[T], List[T]]:
        """Create a batch of items that fits within the context window"""
        pass
    
    @abstractmethod
    def _update_result(self, chain_output: Any) -> Dict[str, Any]:
        """Update the accumulated result with the chain output"""
        pass
    
    async def ainvoke(self, items: List[T], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process items in batches based on context window limitations"""
        # TODO: lets' make this impl more versatile
        # 1. Replace items with a Generic type T that inherits from a pydantic class RefineInput.
        # Thus T = TypeVar('T', bound='RefineInput') and R = TypeVar('R')
        # 2. RefineInput has a field items of type List[Triplets]
        # 3. Refiner should inherit from Runnable[T, R]
        # 3. make _prepare_batch_input should be able to receive batch of triplets or None and T.
        # 4. _prepare_batch_input should be called BEFORE the main loop to produce the very first current_result for sending into _create_batch 
        # 5. Refactor other Refiner children to get rid of duplicate code. 
        # Basically, ainvoke, invoke, _create_batch methods should be only defined in the Refiner class,
        # children classes should only implement _build_chain and _prepare_batch_input.
        # And _update_result should be deleted at all.
        
        current_result = {}
        remaining_items = items.copy()
        batch_idx = 0
        batch_name = self.__class__.__name__
        
        while remaining_items:
            try:
                # Create the next batch based on context window limitations
                current_batch, remaining_items = self._create_batch(remaining_items, current_result)
                
                batch_idx += 1
                logger.info(f"Processing {batch_name} batch {batch_idx} with {len(current_batch)} items. Remaining: {len(remaining_items)}")
                
                # Prepare input for the chain
                chain_input = self._prepare_batch_input(current_batch, current_result)
                
                # Process batch
                with measure_time(f"processing {batch_name} batch {batch_idx}"):
                    result = await self.chain.ainvoke(chain_input)
                
                # Update result with chain output
                current_result = self._update_result(result)
                
            except Exception as e:
                # Handle exceptions
                error_msg = f"Error processing {batch_name} batch {batch_idx}: {str(e)}"
                logger.error(error_msg)
                raise KGConstructionAgentException(error_msg)
        
        return current_result
    
    def invoke(self, items: List[T], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version that runs the async method in an event loop"""
        return asyncio.run(self.ainvoke(items, config))


class OntologyRefiner(Refiner[Dict[str, Any]]):
    """Refiner for constructing and refining an ontology from triplets"""
    
    def _build_chain(self) -> Runnable:
        return build_ontology_refinement_chain(self.llm)
    
    def _prepare_batch_input(self, batch: List[Dict[str, Any]], current_result: Dict[str, Any]) -> Dict[str, Any]:
        formatted_items = "\n".join([
            f"- Subject: {t['triplet'].subject}, Predicate: {t['triplet'].predicate}, Object: {t['triplet'].object}"
            for t in batch
        ])
        
        formatted_result = json.dumps(current_result, indent=2) if current_result else "Empty ontology"
        
        return {
            "triplets": formatted_items,
            "current_ontology": formatted_result
        }
    
    def _create_batch(self, remaining_items: List[Dict[str, Any]], current_result: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        def estimate_triplet_size(triplet):
            t = triplet["triplet"]
            return len(t.subject) + len(t.predicate) + len(t.object) + 10  # +10 for JSON structure overhead

        # Calculate current result size
        result_size = len(json.dumps(current_result)) + 200  # Add buffer
        
        # Dynamically form the next batch based on current result size
        current_batch = []
        current_batch_size = 0
        max_batch_size = (self.context_window_length - result_size) // 2
        
        # Make a copy of remaining items to avoid modifying the original during batch creation
        updated_remaining = remaining_items.copy()
        
        # Add items to the batch until we reach the size limit
        while updated_remaining and current_batch_size < max_batch_size:
            item = updated_remaining[0]
            item_size = estimate_triplet_size(item)
            
            if current_batch_size + item_size <= max_batch_size:
                current_batch.append(item)
                current_batch_size += item_size
                updated_remaining.pop(0)
            else:
                # This item would exceed the batch size limit
                break
        
        # If we couldn't fit any items, take at least one (necessary for progress)
        if not current_batch and updated_remaining:
            current_batch.append(updated_remaining.pop(0))
            
        return current_batch, updated_remaining
    
    def _update_result(self, chain_output: Any) -> Dict[str, Any]:
        return chain_output.ontology


class KnowledgeGraphRefiner(Refiner[Dict[str, Any]]):
    """Refiner for constructing and refining a knowledge graph from triplets and ontology"""
    
    def _build_chain(self) -> Runnable:
        return build_kg_refining_chain(self.llm)
    
    def _prepare_batch_input(self, batch: List[Dict[str, Any]], current_result: Dict[str, Any]) -> Dict[str, Any]:
        # Extract ontology from current_result (it will be stored there during initialization)
        ontology = current_result.get("ontology", {})
        
        formatted_items = "\n".join([
            f"- Subject: {t['triplet'].subject}, Predicate: {t['triplet'].predicate}, Object: {t['triplet'].object}"
            for t in batch
        ])
        
        formatted_ontology = json.dumps(ontology, indent=2) if ontology else "Empty ontology"
        
        formatted_kg = "Empty knowledge graph"
        if current_result.get("entities"):
            formatted_kg = json.dumps(current_result, indent=2)
        
        return {
            "triplets": formatted_items,
            "current_ontology": formatted_ontology,
            "current_kg": formatted_kg
        }
    
    def _create_batch(self, remaining_items: List[Dict[str, Any]], current_result: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        def estimate_triplet_size(triplet):
            t = triplet["triplet"]
            return len(t.subject) + len(t.predicate) + len(t.object) + 10  # +10 for JSON structure overhead

        # Calculate current result size
        ontology = current_result.get("ontology", {})
        result_size = len(json.dumps(current_result)) + len(json.dumps(ontology)) + 200  # Add buffer
        
        # Dynamically form the next batch based on current result size
        current_batch = []
        current_batch_size = 0
        max_batch_size = (self.context_window_length - result_size) // 2
        
        # Make a copy of remaining items to avoid modifying the original during batch creation
        updated_remaining = remaining_items.copy()
        
        # Add items to the batch until we reach the size limit
        while updated_remaining and current_batch_size < max_batch_size:
            item = updated_remaining[0]
            item_size = estimate_triplet_size(item)
            
            if current_batch_size + item_size <= max_batch_size:
                current_batch.append(item)
                current_batch_size += item_size
                updated_remaining.pop(0)
            else:
                # This item would exceed the batch size limit
                break
        
        # If we couldn't fit any items, take at least one (necessary for progress)
        if not current_batch and updated_remaining:
            current_batch.append(updated_remaining.pop(0))
            
        return current_batch, updated_remaining
    
    def _update_result(self, chain_output: Any) -> Dict[str, Any]:
        # Preserve ontology in the result
        current_result = {
            "ontology": chain_output.get("ontology", {}),  # Preserve the ontology
            "entities": chain_output.entities,
            "relationships": chain_output.relationships
        }
        return current_result
    
    async def ainvoke(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process triplets and ontology to create a knowledge graph"""
        # Extract triplets and ontology from the input
        triplets = data.get("triplets", [])
        ontology = data.get("ontology", {})
        
        # Initialize current_result with the ontology
        current_result = {"ontology": ontology}
        
        # Process triplets in batches
        remaining_items = triplets.copy()
        batch_idx = 0
        batch_name = self.__class__.__name__
        
        while remaining_items:
            try:
                # Create the next batch based on context window limitations
                current_batch, remaining_items = self._create_batch(remaining_items, current_result)
                
                batch_idx += 1
                logger.info(f"Processing {batch_name} batch {batch_idx} with {len(current_batch)} items. Remaining: {len(remaining_items)}")
                
                # Prepare input for the chain
                chain_input = self._prepare_batch_input(current_batch, current_result)
                
                # Process batch
                with measure_time(f"processing {batch_name} batch {batch_idx}"):
                    result = await self.chain.ainvoke(chain_input)
                
                # Update result with chain output
                current_result = self._update_result(result)
                
            except Exception as e:
                # Handle exceptions
                error_msg = f"Error processing {batch_name} batch {batch_idx}: {str(e)}"
                logger.error(error_msg)
                raise KGConstructionAgentException(error_msg)
        
        return current_result
    
    def invoke(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version that runs the async method in an event loop"""
        return asyncio.run(self.ainvoke(data, config))


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


    async def triplets_mining(self, state: KGMiningWorkflowState) -> KGMiningWorkflowState:
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
        
        results = await chain.abatch(inputs, return_exceptions=True)
        
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
    
    async def ontology_refining(self, state: KGMiningWorkflowState) -> KGMiningWorkflowState:
        # Check for empty triplets
        if not state.triplets:
            logger.error("No triplets found in state, cannot construct ontology")
            raise KGConstructionAgentException("Cannot construct ontology from empty triplets")
        
        # Use the OntologyRefiner
        refiner = OntologyRefiner(self.llm, self.tokenizer, self.context_window_length)
        current_ontology = await refiner.ainvoke(state.triplets)
        
        # Return updated state with the refined ontology
        return KGMiningWorkflowState(
            chunks=state.chunks,
            triplets=state.triplets,
            ontology=current_ontology,
            knowledge_graph=state.knowledge_graph
        )

    async def kg_refining(self, state: KGMiningWorkflowState) -> KGMiningWorkflowState:
        # Check for empty triplets or ontology
        if not state.triplets:
            logger.error("No triplets found in state, cannot build knowledge graph")
            raise KGConstructionAgentException("Cannot build knowledge graph without triplets")
        
        if not state.ontology:
            logger.error("No ontology found in state, cannot build knowledge graph")
            raise KGConstructionAgentException("Cannot build knowledge graph without ontology")
        
        # Use the KnowledgeGraphRefiner with both triplets and ontology as inputs
        refiner = KnowledgeGraphRefiner(self.llm, self.tokenizer, self.context_window_length)
        current_kg = await refiner.ainvoke({
            "triplets": state.triplets,
            "ontology": state.ontology
        })
        
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


async def run_kg_mining(llm: BaseChatModel, chunks: List[Chunk]) -> KGMiningWorkflowState:
    agent = KGConstructionAgent(llm)
    state = KGMiningWorkflowState(chunks=chunks)
    result = await agent.build_wf().ainvoke(state)
    result = KGMiningWorkflowState.model_validate(result)
    # TODO: save ontology and knowledge graph to file on disk
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
      