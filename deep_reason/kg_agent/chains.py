import asyncio
from itertools import groupby
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
import tiktoken
from transformers import AutoTokenizer
from deep_reason.prompts.kg_prompts import TRIPLETS_PROMPT
from deep_reason.utils import build_chain

from deep_reason.kg_agent.schemes import (
    ChunkTuple, TripletList, OntologyStructure, KgStructure, AggregationInput
)
from deep_reason.kg_agent.utils import AggregationHelper, logger, measure_time
from deep_reason.schemes import Chunk


logger = logging.getLogger(__name__)


def build_triplets_mining_chain(llm: BaseChatModel) -> Runnable:
    system_template = TRIPLETS_PROMPT

    human_template = """Extract knowledge triplets from the following text chunk:
    
Current chunk: {current_chunk}

{left_context_prefix}{left_context}

{right_context_prefix}{right_context}

{response_format_description}"""
    
    parser = PydanticOutputParser(pydantic_object=TripletList)
    
    return build_chain(llm, system_template, human_template, parser)


def build_ontology_refinement_chain(llm: BaseChatModel) -> Runnable['AggregationInput', Dict[str, Any]]:
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
    {triplet_list}

    Current Ontology:
    {current_ontology}

    Create or refine the ontology to categorize entity types and relationship types.
    {response_format_description}"""
    
    parser = PydanticOutputParser(pydantic_object=OntologyStructure)
    
    chain = build_chain(llm, system_template, human_template, parser)
    
    def _process_input(refiner_input: AggregationInput):
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


def build_kg_refining_chain(llm: BaseChatModel) -> Runnable['AggregationInput', Dict[str, Any]]:
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
    
    def _process_input(refiner_input: AggregationInput):
        from deep_reason.kg_agent.utils import KGConstructionAgentException

        ontology = refiner_input.input.get("ontology", None)
        if not ontology:
            raise KGConstructionAgentException("No ontology provided for kg refining chain")
        
        current_graph = refiner_input.input.get("current_graph", None)
        if not current_graph:
            logger.info("No current graph provided, creating empty graph")
            current_graph = KgStructure(kg_nodes=[], kg_triplets=[])

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


def build_kg_refining_map_chain(llm: BaseChatModel) -> Runnable['AggregationInput', Dict[str, Any]]:
    chain = build_kg_refining_chain(llm=llm)

    def _process_output(x: Dict[str, Any]) -> Dict[str, Any]:
        return {"current_graph": x["current_graph"]}
    
    return(chain | RunnableLambda(_process_output))


async def reduce_partial_kg(partial_kgs: List[Dict[str, Any]]) -> Dict[str, Any]:
    kg_nodes = []
    kg_triplets = []
    for partial_kg in partial_kgs:
        partial_kg_structure = cast(KgStructure, partial_kg['current_graph']) 
        kg_nodes.extend(partial_kg_structure.kg_nodes)
        kg_triplets.extend(partial_kg_structure.kg_triplets)
    
    return {"current_graph": KgStructure(kg_nodes=kg_nodes, kg_triplets=kg_triplets)}


class Refiner(AggregationHelper, Runnable[AggregationInput, Dict[str, Any]]):
    """Abstract class for refining data in batches with LLM chains"""

    def __init__(self, refine_chain: Runnable[AggregationInput, Dict[str, Any]], tokenizer=None, context_window_length: int = 8000):
        self.refine_chain = refine_chain
        self.context_window_length = context_window_length
        if tokenizer is None:
            # default tokenizer if nothing provided
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    async def ainvoke(self, input: AggregationInput, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process items in batches based on context window limitations"""
        from deep_reason.kg_agent.utils import KGConstructionAgentException

        current_state = input.input
        remaining_items = input.items.copy()
        batch_idx = 0
        batch_name = self.__class__.__name__

        while remaining_items:
            try:
                # Create the next batch based on context window limitations
                current_batch, remaining_items = self._create_batch(remaining_items, current_state)

                batch_idx += 1
                logger.info(f"Processing {batch_name} batch {batch_idx} with {len(current_batch.items)} items. Remaining: {len(remaining_items)}")

                # Process batch
                with measure_time(f"processing {batch_name} batch {batch_idx}"):
                    current_state = await self.refine_chain.ainvoke(current_batch, config=config)

            except Exception as e:
                # Handle exceptions
                error_msg = f"Error processing {batch_name} batch {batch_idx}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise KGConstructionAgentException(error_msg)

        return current_state

    def invoke(self, input: AggregationInput, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version that runs the async method in an event loop"""
        return asyncio.run(self.ainvoke(input, config))


class MapReducer(AggregationHelper, Runnable[AggregationInput, Dict[str, Any]]):
    """Process items by first mapping them individually and then reducing the results."""

    def __init__(self,
                 map_chain: Runnable[AggregationInput, Dict[str, Any]],
                 reduce_chain: Runnable[List[Dict[str, Any]], Dict[str, Any]] | Callable[[List[Dict[str, Any]]],Awaitable[Dict[str, Any]]],
                 tokenizer=None,
                 context_window_length: int = 8000):
        """Initialize the MapReducer.

        Args:
            map_chain: Chain to apply to each item or batch of items
            reduce_chain: Chain to combine results from map phase
            tokenizer: Tokenizer to use for estimating token counts
            context_window_length: Maximum context window size in tokens
        """
        self.map_chain = map_chain
        self.reduce_chain = reduce_chain
        self.context_window_length = context_window_length
        if tokenizer is None:
            # default tokenizer if nothing provided
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    async def ainvoke(self, input: AggregationInput, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process items using map-reduce pattern with parallel mapping."""
        from deep_reason.kg_agent.utils import KGConstructionAgentException

        items = input.items
        initial_state = input.input

        if not items:
            return initial_state

        # Create batches for parallel processing
        batches = []
        current_items = items.copy()
        while current_items:
            batch, current_items = self._create_batch(current_items, initial_state)
            batches.append(batch)

        logger.info(f"Created {len(batches)} batches for parallel processing")

        # Map phase: Process all batches in parallel
        with measure_time("parallel map phase"):
            map_results = await self.map_chain.abatch(batches, config=config)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(map_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {i}: {result}")
            else:
                valid_results.append(result)

        if not valid_results:
            logger.error("No valid results from map phase")
            raise KGConstructionAgentException("No valid results from map phase")

        # Reduce phase: Combine all results
        logger.info(f"Starting reduce phase with {len(valid_results)} results")

        try:
            with measure_time("processing reduce phase"):
                # Process with reduce chain
                if isinstance(self.reduce_chain, Runnable):
                    final_result = await self.reduce_chain.ainvoke(valid_results, config=config)
                else:
                    final_result = await self.reduce_chain(valid_results)

        except Exception as e:
            error_msg = f"Error in reduce phase: {str(e)}"
            logger.error(error_msg)
            raise KGConstructionAgentException(error_msg)

        return final_result

    def invoke(self, input: AggregationInput, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version that runs the async method in an event loop."""
        return asyncio.run(self.ainvoke(input, config))


class TripletsMiner(Runnable):
    """Process chunks to extract knowledge triplets using sliding window context approach."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    async def ainvoke(self, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Process chunks to extract triplets with surrounding context.
        
        Args:
            chunks: List of chunks to process
            config: Optional configuration for the chain
            
        Returns:
            List of dictionaries containing chunk and triplet information
        """
        logger.info(f"Sorting chunks by document_id and order_id for {len(chunks)} chunks")
        # 1. Order chunks by document_id and order_id
        chunks = sorted(chunks, key=lambda x: (x.document_id, x.order_id))
        
        # 2. Create chunk tuples with sliding window approach
        chunk_tuples = []
        logger.info(f"Creating chunks tuples by document_id and order_id for {len(chunks)} chunks")
        # Group chunks by document_id
        for _, doc_chunks in groupby(chunks, key=lambda x: x.document_id):
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
        logger.info(f"Forming batches for {len(chunk_tuples)} chunk tuples")
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
        
        logger.info(f"Processing {len(inputs)} inputs")
        results = await chain.abatch(inputs, return_exceptions=True, config=config)
        
        # 5. Handle exceptions
        valid_results = []
        for chunk_tuple, result in zip(chunk_tuples, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {chunk_tuple.current_chunk.chapter_name}: {result}. Skipping chunk.")
                continue
            # Attach the chunk information to each result
            for triplet in result.triplets:
                valid_results.append(
                    (chunk_tuple.current_chunk.document_id, chunk_tuple.current_chunk.order_id, triplet)
                )
        
        # 6. Sort triplets by document and then chunk order
        valid_results = [triplet for _, _, triplet in sorted(valid_results, key=lambda x: (x[0], x[1]))]
        
        return valid_results
    
    def invoke(self, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Synchronous version that runs the async method in an event loop."""
        return asyncio.run(self.ainvoke(chunks, config))

