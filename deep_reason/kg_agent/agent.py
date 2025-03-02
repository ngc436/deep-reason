import logging
import os
import asyncio
from typing import Any, Dict, List, Optional
from itertools import groupby

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END
from transformers import PreTrainedTokenizerBase

from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.schemes import Chunk
from deep_reason.utils import VLLMChatOpenAI
from deep_reason.kg_agent.schemes import (
    AggregationInput, KGMiningWorkflowState, ChunkTuple, KGMiningResult
)
from deep_reason.kg_agent.utils import KGConstructionAgentException, load_obliqa_dataset
from deep_reason.kg_agent.chains import (
    MapReducer, Refiner, build_triplets_mining_chain, build_ontology_refinement_chain, 
    build_kg_refining_chain, build_kg_refining_map_chain, reduce_partial_kg
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)", 
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class KGConstructionAgent:
    def __init__(self, 
                 llm: BaseChatModel, 
                 tokenizer: Optional[PreTrainedTokenizerBase | str] = None, 
                 context_window_length: int = 8000, 
                 use_refine_for_kg: bool = False):
        self.llm = llm
        self.context_window_length = context_window_length
        self.tokenizer = tokenizer
        self.use_refine_for_kg = use_refine_for_kg

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
        refiner_input = AggregationInput(items=state.triplets, input={"current_ontology": None})
        result = await refiner.ainvoke(input=refiner_input, config=config)
        current_ontology = result["current_ontology"]

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
        
        if self.use_refine_for_kg:
            chain = Refiner(
                refine_chain=build_kg_refining_chain(self.llm), 
                tokenizer=self.tokenizer, 
                context_window_length=self.context_window_length
            )
        else:
            chain = MapReducer(
                map_chain=build_kg_refining_map_chain(self.llm),
                reduce_chain=reduce_partial_kg,
                tokenizer=self.tokenizer,
                context_window_length=self.context_window_length
            )

        agg_input = AggregationInput(
            items=state.triplets,
            input={
                "ontology": state.ontology,
                "current_graph": None
            }
        )
        result = await chain.ainvoke(input=agg_input, config=config)
        current_kg = result["current_graph"]
        
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
