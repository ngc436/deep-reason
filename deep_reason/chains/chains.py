from contextlib import contextmanager
from datetime import datetime
import json
import logging
import pandas as pd
import os
from typing import Any, Dict, List, Optional, cast, Union
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

from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.schemes import Chunk
from deep_reason.utils import VLLMChatOpenAI
from examples.kg_extraction import load_obliqa_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)", 
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class KGMiningWorkflowState(BaseModel):
    chunks: List[Chunk]
    triplets: List[Dict[str, Any]] = Field(default_factory=list)
    ontology: Dict[str, List[str]] = Field(default_factory=dict)
    knowledge_graph: List[Dict[str, Any]] = Field(default_factory=list)


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

    class Triplet(BaseModel):
        subject: str = Field(description="The entity performing the action or having the property")
        predicate: str = Field(description="The relationship or action")
        object: str = Field(description="The entity receiving the action or the value of the property")
    
    class TripletList(BaseModel):
        triplets: List[Triplet] = Field(description="List of knowledge triplets extracted from the text")
    
    parser = PydanticOutputParser(pydantic_object=TripletList)
    
    return build_chain(llm, system_template, human_template, parser)


def build_ontology_and_kg_compiling_chain(llm: BaseChatModel) -> Runnable:
    system_template = """You are an expert knowledge graph engineer. Your task is to organize extracted knowledge triplets into:
1. An ontology - categorizing entities and relationships
2. A structured knowledge graph

Process the given triplets and incrementally build or refine the existing ontology and knowledge graph."""

    human_template = """Process the following knowledge triplets:

{triplets}

Current Ontology (entities and relationships organized by category):
{current_ontology}

Current Knowledge Graph:
{current_kg}

Please analyze these triplets and update both the ontology and knowledge graph.
{response_format_description}"""

    class Entity(BaseModel):
        entity_id: str = Field(description="Unique identifier for the entity")
        name: str = Field(description="Name or label of the entity")
        category: str = Field(description="Category or type of the entity")

    class Relationship(BaseModel):
        source_id: str = Field(description="Entity ID of the source/subject")
        relation_type: str = Field(description="Type of relationship/predicate")
        target_id: str = Field(description="Entity ID of the target/object")
        confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)

    class OntologyAndKG(BaseModel):
        ontology: Dict[str, List[str]] = Field(description="Categories of entities and relationships")
        entities: List[Entity] = Field(description="List of entities in the knowledge graph")
        relationships: List[Relationship] = Field(description="List of relationships in the knowledge graph")
    
    parser = PydanticOutputParser(pydantic_object=OntologyAndKG)
    
    return build_chain(llm, system_template, human_template, parser)


class KGConstructionAgent:
    def __init__(self, llm: BaseChatModel, context_window_length: int = 8000):
        self.llm = llm
        self.context_window_length = context_window_length

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
        
        # 6 & 7. Create and return new state with valid results
        return KGMiningWorkflowState(
            chunks=state.chunks,
            triplets=valid_results,
            ontology=state.ontology,
            knowledge_graph=state.knowledge_graph
        )

    async def ontology_and_kg_compiling(self, state: KGMiningWorkflowState) -> KGMiningWorkflowState:
        if not state.triplets:
            logger.warning("No triplets found in state, skipping ontology and kg compilation")
            return state
        
        # Build the chain for ontology and KG compilation
        chain = build_ontology_and_kg_compiling_chain(self.llm)
        
        # Initialize empty ontology and knowledge graph
        current_ontology = {}
        current_kg = {"entities": [], "relationships": []}
        
        # Estimate triplet size for batching (rough approximation)
        def estimate_triplet_size(triplet):
            t = triplet["triplet"]
            return len(t.subject) + len(t.predicate) + len(t.object) + 10  # +10 for JSON structure overhead
        
        # Create batches based on context window
        batches = []
        current_batch = []
        current_batch_size = 0
        
        # Sort triplets by document and then chunk order
        sorted_triplets = sorted(
            state.triplets, 
            key=lambda x: (x["chunk"].document_id, x["chunk"].order_id)
        )
        
        # Create batches
        for triplet in sorted_triplets:
            triplet_size = estimate_triplet_size(triplet)
            
            # If adding this triplet would exceed context window, start new batch
            if current_batch_size + triplet_size > self.context_window_length // 2:  # Half context for triplets, half for other data
                if current_batch:  # Only append if there are triplets in the batch
                    batches.append(current_batch)
                current_batch = [triplet]
                current_batch_size = triplet_size
            else:
                current_batch.append(triplet)
                current_batch_size += triplet_size
        
        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} batches of triplets for processing")
        
        # Process batches iteratively, refining the ontology and knowledge graph
        for batch_idx, batch in enumerate(batches):
            try:
                # Format triplets for the chain
                triplets_text = "\n".join([
                    f"- Subject: {t['triplet'].subject}, Predicate: {t['triplet'].predicate}, Object: {t['triplet'].object}"
                    for t in batch
                ])
                
                # Format current ontology and KG
                current_ontology_text = json.dumps(current_ontology, indent=2) if current_ontology else "Empty ontology"
                current_kg_text = json.dumps(current_kg, indent=2) if current_kg.get("entities") or current_kg.get("relationships") else "Empty knowledge graph"
                
                # Prepare input for the chain
                chain_input = {
                    "triplets": triplets_text,
                    "current_ontology": current_ontology_text,
                    "current_kg": current_kg_text
                }
                
                # Process batch
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
                with measure_time(f"processing batch {batch_idx+1}"):
                    result = await chain.ainvoke(chain_input)
                
                # Update ontology and knowledge graph with results
                current_ontology = result.ontology
                current_kg = {
                    "entities": result.entities,
                    "relationships": result.relationships
                }
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                raise Exception(f"Failed to process triplets batch {batch_idx+1}: {str(e)}")
        
        # Return updated state
        return KGMiningWorkflowState(
            chunks=state.chunks,
            triplets=state.triplets,
            ontology=current_ontology,
            knowledge_graph=current_kg
        )

    def build_wf(self) -> Runnable[KGMiningWorkflowState, Dict[str, Any]]:
        wf = StateGraph(KGMiningWorkflowState)
        
        wf.add_node("triplets_mining", self.triplets_mining)
        wf.add_node("ontology_and_kg_compiling", self.ontology_and_kg_compiling)

        wf.add_edge(START, "triplets_mining")
        wf.add_edge("triplets_mining", "ontology_and_kg_compiling")
        wf.add_edge("ontology_and_kg_compiling", END)

        return wf.compile()


async def run_kg_mining(llm: BaseChatModel, chunks: List[Chunk]) -> KGMiningWorkflowState:
    agent = KGConstructionAgent(llm)
    state = KGMiningWorkflowState(chunks=chunks)
    result = await agent.build_wf().ainvoke(state)
    result = KGMiningWorkflowState.model_validate(result)
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
      