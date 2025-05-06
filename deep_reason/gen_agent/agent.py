import os
import json
from typing import Dict, List, Tuple, Any, TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from tqdm.asyncio import tqdm
from datetime import datetime
from langgraph.graph import StateGraph, START, END

from deep_reason.gen_agent.sampling import (
    optimized_extract_entity_chains,
    map_entities_to_descriptions,
    extract_chain_relationships
)
from deep_reason.gen_agent.prompts import COMPLEX_RELATIONSHIPS_PROMPT, PREPARE_FOR_KNOWLEDGE_EDITING_PROMPT


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
    subject: str = Field(description="The subject of the edit prompt")
    target: str = Field(description="The actual answer to the edit prompt that is one of the entities")
    generalization_prompt: str = Field(description="The generalization prompt to check that editing is successful with a bit changed edit prompt")
    locality_prompt: str = Field(description="The locality prompt to check that model does not influenced on unrelated to editing inputs (though connected by entity)")
    portability_prompt: str = Field(description="The portability prompt to measure success for reasoning or application")


class AgentState(TypedDict):
    """State for the ComplexRelationshipAgent workflow"""
    entity_chain: List[str]
    entity_descriptions: Dict[str, Dict[str, Any]]
    relationships: Dict[Tuple[str, str], Dict[str, Any]]
    complex_relationship: Optional[ComplexRelationship]
    knowledge_editing_input: Optional[KnowledgeEditingInput]

class ComplexRelationshipAgent:
    """Agent for inferring complex relationships between chain endpoints"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        graphml_path: str,
        entities_parquet_path: str,
        relationships_parquet_path: str,
        chain_length: int = 3,
        n_samples: int = 15,
        max_retries: int = 3,
        output_dir: str = "results",
        max_concurrency: int = 100
    ):
        self.llm = llm
        self.graphml_path = graphml_path
        self.entities_parquet_path = entities_parquet_path
        self.relationships_parquet_path = relationships_parquet_path
        self.chain_length = chain_length
        self.n_samples = n_samples
        self.max_retries = max_retries
        self.output_dir = output_dir
        self.max_concurrency = max_concurrency
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for relationship inference and knowledge editing input preparation"""
        
        # Create the output parser for complex relationships
        relationship_parser = PydanticOutputParser(pydantic_object=ComplexRelationship)
        
        # Create the output parser for knowledge editing input
        editing_parser = PydanticOutputParser(pydantic_object=KnowledgeEditingInput)
        
        # Create the prompt templates
        self.relationship_prompt = PromptTemplate(
            input_variables=["entity_chain", "entity_descriptions", "relationships"],
            template=COMPLEX_RELATIONSHIPS_PROMPT,
            partial_variables={"schema": relationship_parser.get_format_instructions()}
        )
        
        self.editing_prompt = PromptTemplate(
            input_variables=["entities", "relationships", "descriptions"],
            template=PREPARE_FOR_KNOWLEDGE_EDITING_PROMPT,
            partial_variables={"schema": editing_parser.get_format_instructions()}
        )
        
        # Build the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step
        workflow.add_node("infer_relationship", self._infer_relationship_node)
        workflow.add_node("prepare_editing_input", self._prepare_editing_input_node)
        
        # Add edges
        workflow.add_edge(START, "infer_relationship")
        workflow.add_edge("infer_relationship", "prepare_editing_input")
        workflow.add_edge("prepare_editing_input", END)
        
        return workflow.compile()
    
    async def _infer_relationship_node(self, state: AgentState) -> AgentState:
        """Node for inferring complex relationships"""
        # Prepare input for the relationship inference
        input_data = {
            "entity_chain": self._format_entity_chain(state["entity_chain"]),
            "entity_descriptions": self._format_entity_descriptions(state["entity_descriptions"]),
            "relationships": self._format_relationships(state["relationships"])
        }
        
        # Run the relationship inference
        result = await self._parse_relationship_output_with_retry(
            await self.llm.ainvoke(
                self.relationship_prompt.format(**input_data)
            )
        )
        
        return {**state, "complex_relationship": result}
    
    async def _prepare_editing_input_node(self, state: AgentState) -> AgentState:
        """Node for preparing knowledge editing input"""
        # Prepare input for knowledge editing input preparation
        input_data = {
            "entities": f"{state['entity_chain'][0]}, {state['entity_chain'][-1]}",
            "relationships": self._format_relationships(state["relationships"]),
            "descriptions": self._format_entity_descriptions(state["entity_descriptions"])
        }
        
        # Run the knowledge editing input preparation
        result = await self._parse_editing_output_with_retry(
            await self.llm.ainvoke(
                self.editing_prompt.format(**input_data)
            )
        )
        
        return {**state, "knowledge_editing_input": result}
    
    def _format_entity_descriptions(self, descriptions: Dict[str, Dict[str, Any]]) -> str:
        """Format entity descriptions for the prompt"""
        formatted = []
        for entity, desc in descriptions.items():
            formatted.append(f"{entity}: {desc['description']}")
        return "\n".join(formatted)
    
    def _format_relationships(self, relationships: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
        """Format relationships for the prompt"""
        formatted = []
        for (source, target), rel in relationships.items():
            formatted.append(f"{source} -> {target}: {rel['description']}")
        return "\n".join(formatted)
    
    def _format_entity_chain(self, chain: List[str]) -> str:
        """Format entity chain with arrows between entities
        
        Args:
            chain: List of entities in the chain
            
        Returns:
            Formatted string with entities joined by arrows
        """
        return " -> ".join(chain)
    
    async def _parse_editing_output_with_retry(self, output: Any) -> KnowledgeEditingInput:
        """Parse the LLM output for knowledge editing input with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Extract content from AIMessage if needed
                if hasattr(output, 'content'):
                    output = output.content
                
                # Try to parse as JSON first
                json_str = output.strip()
                if not json_str.startswith('{'):
                    # If not starting with {, try to find JSON in the output
                    start = json_str.find('{')
                    end = json_str.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = json_str[start:end]
                
                data = json.loads(json_str)
                return KnowledgeEditingInput(**data)
            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                if attempt < self.max_retries - 1:
                    # If parsing fails, ask the LLM to fix the output
                    retry_prompt = ChatPromptTemplate.from_template(
                        """
                        The previous response was not valid JSON. Please fix it to match this schema:
                        {schema}
                        
                        Previous response:
                        {output}
                        
                        Please provide a valid JSON response that strictly follows the schema.
                        """
                    )
                    retry_chain = retry_prompt | self.llm
                    output = await retry_chain.ainvoke({
                        "schema": KnowledgeEditingInput.model_json_schema(),
                        "output": output
                    })
                    continue
                else:
                    raise ValueError(f"Failed to parse knowledge editing input after {self.max_retries} attempts")
    
    async def _parse_relationship_output_with_retry(self, output: Any) -> ComplexRelationship:
        """Parse the LLM output for complex relationships with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Extract content from AIMessage if needed
                if hasattr(output, 'content'):
                    output = output.content
                
                # Try to parse as JSON first
                json_str = output.strip()
                if not json_str.startswith('{'):
                    # If not starting with {, try to find JSON in the output
                    start = json_str.find('{')
                    end = json_str.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = json_str[start:end]
                
                data = json.loads(json_str)
                return ComplexRelationship(**data)
            except (json.JSONDecodeError, ValueError, AttributeError) as e:
                if attempt < self.max_retries - 1:
                    # If parsing fails, ask the LLM to fix the output
                    retry_prompt = ChatPromptTemplate.from_template(
                        """
                        The previous response was not valid JSON. Please fix it to match this schema:
                        {schema}
                        
                        Previous response:
                        {output}
                        
                        Please provide a valid JSON response that strictly follows the schema.
                        """
                    )
                    retry_chain = retry_prompt | self.llm
                    output = await retry_chain.ainvoke({
                        "schema": ComplexRelationship.model_json_schema(),
                        "output": output
                    })
                    continue
                else:
                    # If all retries fail, return a default object
                    return ComplexRelationship(
                        first_entity="",
                        last_entity="",
                        relationships=["no_relationship"],
                        evidence=[]
                    )
    
    async def infer_relationships(self) -> List[Dict[str, Any]]:
        """Infer complex relationships for sampled chains and prepare knowledge editing inputs"""
        
        # Sample chains
        print("Sampling entity chains...")
        chains = list(optimized_extract_entity_chains(
            self.graphml_path,
            self.chain_length,
            self.n_samples
        ))
        
        # Get entity descriptions
        print("Mapping entities to descriptions...")
        entity_descriptions = map_entities_to_descriptions(
            chains,
            self.entities_parquet_path
        )
        
        # Get relationships
        print("Extracting chain relationships...")
        relationships = extract_chain_relationships(
            chains,
            self.relationships_parquet_path
        )
        
        # Process each chain with progress bar
        print("Processing chains to infer relationships and prepare knowledge editing inputs...")
        results = []
        
        # Process chains in batches
        for i in range(0, len(chains), self.max_concurrency):
            batch = chains[i:i + self.max_concurrency]
            batch_results = []
            
            async for chain in tqdm(batch, desc=f"Processing batch {i//self.max_concurrency + 1}"):
                try:
                    # Prepare initial state
                    state = {
                        "entity_chain": chain,
                        "entity_descriptions": entity_descriptions,
                        "relationships": relationships,
                        "complex_relationship": None,
                        "knowledge_editing_input": None
                    }
                    
                    # Run the workflow
                    final_state = await self.workflow.ainvoke(state)
                    
                    # Convert to result format
                    result_dict = {
                        "chain": list(chain),
                        "first_entity": final_state["complex_relationship"].first_entity,
                        "last_entity": final_state["complex_relationship"].last_entity,
                        "entity_descriptions": {
                            entity: desc["description"]
                            for entity, desc in entity_descriptions.items()
                            if entity in chain
                        },
                        "relationship_descriptions": [
                            f"{source} -> {target}: {rel['description']}"
                            for (source, target), rel in relationships.items()
                            if source in chain and target in chain
                        ],
                        "inferred_relationships": final_state["complex_relationship"].relationships,
                        "evidence": final_state["complex_relationship"].evidence,
                        "knowledge_editing_input": final_state["knowledge_editing_input"].model_dump()
                    }
                    
                    batch_results.append(result_dict)
                except Exception as e:
                    print(f"Error processing chain {chain}: {str(e)}")
                    # Add a default result for failed chains
                    result_dict = {
                        "chain": list(chain),
                        "first_entity": chain[0],
                        "last_entity": chain[-1],
                        "entity_descriptions": {},
                        "relationship_descriptions": [],
                        "inferred_relationships": ["no_relationship"],
                        "evidence": [],
                        "knowledge_editing_input": None
                    }
                    batch_results.append(result_dict)
            
            results.extend(batch_results)
        
        # Save results to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"complex_agent_result_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        
        return results
