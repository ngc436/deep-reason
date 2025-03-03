import logging
import os
import asyncio
from typing import Any, Dict, List, Optional, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, START, END
from transformers import PreTrainedTokenizerBase

from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.schemes import Chunk, OntologyStructure, Triplet
from deep_reason.utils import VLLMChatOpenAI
from deep_reason.kg_agent.schemes import (
    AggregationInput, KGMiningWorkflowState, KGMiningResult, KgStructure, TripletList
)
from deep_reason.kg_agent.utils import KGConstructionAgentException, load_obliqa_dataset, CacheManager
from deep_reason.kg_agent.chains import (
    MapReducer, Refiner, build_ontology_refinement_chain, 
    build_kg_refining_chain, build_kg_refining_map_chain, reduce_partial_kg, TripletsMiner
)


logger = logging.getLogger(__name__)


class KGConstructionAgent:
    def __init__(self, 
                 llm: BaseChatModel, 
                 tokenizer: Optional[PreTrainedTokenizerBase | str] = None, 
                 context_window_length: int = 8000, 
                 use_refine_for_kg: bool = False,
                 cache_dir: str = ".cache"):
        self.llm = llm
        self.context_window_length = context_window_length
        self.tokenizer = tokenizer
        self.use_refine_for_kg = use_refine_for_kg
        
        # Create a single cache manager instance
        self.cache_manager = CacheManager(cache_dir=cache_dir)

    async def _triplets_mining(self, state: KGMiningWorkflowState, config: Optional[Dict[str, Any]] = None) -> KGMiningWorkflowState:
        logger.info(f"Mining triplets for {len(state.chunks)} chunks")
        
        # Check cache first
        cached_result = self.cache_manager.get(state.chunks, TripletList, prefix="triplets")
        if cached_result and not state.no_cache:
            logger.info(f"Using cached triplets mining result")
            return state.model_copy(update={"triplets": cached_result.triplets})
            
        # Use the TripletsMiner class to extract triplets from chunks
        miner = TripletsMiner(self.llm)
        valid_results = await miner.ainvoke(state.chunks, config=config)
        logger.info(f"Finished mining triplets. Found {len(valid_results)} triplets")
        
        # Cache the result
        self.cache_manager.put(state.chunks, TripletList(triplets=valid_results), prefix="triplets")
        
        return state.model_copy(update={"triplets": valid_results})
    
    async def _ontology_refining(self, state: KGMiningWorkflowState, config: Optional[Dict[str, Any]] = None) -> KGMiningWorkflowState:
        logger.info(f"Refining ontology for {len(state.triplets)} triplets")
        
        # Check if we have empty triplets
        if not state.triplets:
            logger.error("No triplets found in state, cannot construct ontology")
            raise KGConstructionAgentException("Cannot construct ontology from empty triplets")
        
        # Check cache first
        cached_ontology = None #self.cache_manager.get(state.triplets, OntologyStructure, prefix="ontology")
        if cached_ontology and not state.no_cache:
            logger.info(f"Using cached ontology refining result")
            return state.model_copy(update={"ontology": cached_ontology})
        
        refiner = Refiner(
            refine_chain=build_ontology_refinement_chain(self.llm), 
            tokenizer=self.tokenizer, 
            context_window_length=self.context_window_length
        )
        refiner_input = AggregationInput(items=state.triplets, input={"current_ontology": None})
        result = await refiner.ainvoke(input=refiner_input, config=config)
        current_ontology = result["current_ontology"]
        logger.info(f"Finished refining ontology. Found {len(current_ontology.nodes)} nodes and {len(current_ontology.relations)} relations")
        
        # Cache the result
        # self.cache_manager.put(state.triplets, current_ontology, prefix="ontology")
        
        return state.model_copy(update={"ontology": current_ontology})

    async def _kg_refining(self, state: KGMiningWorkflowState, config: Optional[Dict[str, Any]] = None) -> KGMiningWorkflowState:
        logger.info(f"Refining knowledge graph for {len(state.triplets)} triplets")
        
        # Check for empty triplets or ontology
        if not state.triplets:
            logger.error("No triplets found in state, cannot build knowledge graph")
            raise KGConstructionAgentException("Cannot build knowledge graph without triplets")
        
        if not state.ontology:
            logger.error("No ontology found in state, cannot build knowledge graph")
            raise KGConstructionAgentException("Cannot build knowledge graph without ontology")
        
        # Check cache first - use both triplets and ontology as key
        cache_key = {"triplets": state.triplets, "ontology": state.ontology}
        cached_graph = None #self.cache_manager.get(cache_key, KgStructure, prefix="kg")
        if cached_graph and not state.no_cache:
            logger.info(f"Using cached knowledge graph refining result")
            return state.model_copy(update={"knowledge_graph": cached_graph})
        
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
        logger.info(f"Finished refining knowledge graph. Found {len(current_kg.kg_nodes)} nodes and {len(current_kg.kg_triplets)} triplets")
        
        # Cache the result
        # self.cache_manager.put(cache_key, current_kg, prefix="kg")
        
        return state.model_copy(update={"knowledge_graph": current_kg})

    def build_wf(self) -> Runnable[KGMiningWorkflowState, Dict[str, Any]]:
        wf = StateGraph(KGMiningWorkflowState)
        
        wf.add_node("triplets_mining", self._triplets_mining)
        wf.add_node("ontology_refining", self._ontology_refining)
        wf.add_node("kg_refining", self._kg_refining)

        wf.add_edge(START, "triplets_mining")
        wf.add_edge("triplets_mining", "ontology_refining")
        wf.add_edge("ontology_refining", "kg_refining")
        wf.add_edge("kg_refining", END)

        return wf.compile()


async def run_kg_mining(llm: BaseChatModel, chunks: List[Chunk], output_path: str = "kg_output"):
    # alternative for debuggin purposes
    # import pickle
    # valide_results_path = "/tmp/valid_results.pickle"
    # if not os.path.exists(valide_results_path):
    #     miner = TripletsMiner(llm)
    #     valid_results = await miner.ainvoke(chunks[:10])
    #     with open(valide_results_path, "wb") as f:
    #         pickle.dump(valid_results, f)
    # else:
    #     with open(valide_results_path, "rb") as f:
    #         valid_results = pickle.load(f)

    # ontology_path = "/tmp/ontology.pickle"
    # if not os.path.exists(ontology_path):
    #     refiner = Refiner(
    #         refine_chain=build_ontology_refinement_chain(llm), 
    #         tokenizer=None, 
    #         context_window_length=25_000
    #     )
    #     refiner_input = AggregationInput(items=valid_results, input={"current_ontology": None})
    #     result = await refiner.ainvoke(input=refiner_input, config=None)
    #     current_ontology = result["current_ontology"]
    #     with open(ontology_path, "wb") as f:
    #         pickle.dump(current_ontology, f)
    # else:
    #     with open(ontology_path, "rb") as f:
    #         current_ontology = pickle.load(f)

    # kg_path = "/tmp/kg.pickle"
    # if not os.path.exists(kg_path):
    #     chain = MapReducer(
    #         map_chain=build_kg_refining_map_chain(llm),
    #         reduce_chain=reduce_partial_kg,
    #         tokenizer=None,
    #         context_window_length=25_000
    #     )
    #     agg_input = AggregationInput(
    #         items=valid_results,
    #         input={
    #             "ontology": current_ontology,
    #             "current_graph": None
    #         }
    #     )
    #     result = await chain.ainvoke(input=agg_input, config=None)
    #     current_kg = result["current_graph"]
    #     with open(kg_path, "wb") as f:  
    #         pickle.dump(current_kg, f)
    # else:
    #     with open(kg_path, "rb") as f:
    #         current_kg = pickle.load(f)

    agent = KGConstructionAgent(llm)
    state = KGMiningWorkflowState(chunks=chunks, no_cache=False)
    result = await agent.build_wf().ainvoke(state, config={"max_concurrency": 100})
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


def main():
    llm = VLLMChatOpenAI(
        model="/model",
        base_url=os.environ[OPENAI_API_BASE],
        api_key=os.environ[OPENAI_API_KEY],
        temperature=0.3,
        max_tokens=8096
    )

    chunks = load_obliqa_dataset(obliqa_dir="datasets/ObliQA/StructuredRegulatoryDocuments")

    asyncio.run(run_kg_mining(llm, chunks[:10]))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()

