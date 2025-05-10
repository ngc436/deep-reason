import os
import json
from typing import Dict, List, Tuple, Any, TypedDict, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from tqdm.asyncio import tqdm
from datetime import datetime
from langgraph.graph import StateGraph, START, END

from deep_reason.gen_agent.sampling import (
    optimized_extract_entity_chains,
    optimized_extract_community_chains,
    map_entities_to_descriptions,
    extract_chain_relationships
)
from deep_reason.gen_agent.prompts import COMPLEX_RELATIONSHIPS_PROMPT, PREPARE_FOR_KNOWLEDGE_EDITING_PROMPT
from deep_reason.gen_agent.scheme import (
    ComplexRelationship,
    ComplexRelationshipResult,
    KnowledgeEditingInput,
    InferredRelationship,
    Generalization,
    Locality,
    Portability
)


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
        max_concurrency: int = 100,
        use_communities: bool = False,
        communities_parquet_path: Optional[str] = None,
        n_communities: Optional[int] = None,
        n_samples_per_community: Optional[int] = None,
        dataset_name: str = "unknown"
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
        self.use_communities = use_communities
        self.communities_parquet_path = communities_parquet_path
        self.n_communities = n_communities
        self.n_samples_per_community = n_samples_per_community
        self.dataset_name = dataset_name
        
        # Validate community-related parameters
        if self.use_communities:
            if not self.communities_parquet_path:
                raise ValueError("communities_parquet_path must be provided when use_communities is True")
            if not self.n_communities:
                raise ValueError("n_communities must be provided when use_communities is True")
            # n_samples_per_community can be None to get all possible chains
        
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
        """Node for preparing knowledge editing input for each inferred relationship"""
        complex_relationship = state.get("complex_relationship")
        if not complex_relationship or not complex_relationship.relationships:
            # Fallback: behave as before if no relationships
            input_data = {
                "entities": f"{state['entity_chain'][0]}, {state['entity_chain'][-1]}",
                "relationships": self._format_relationships(state["relationships"]),
                "descriptions": self._format_entity_descriptions(state["entity_descriptions"])
            }
            result = await self._parse_editing_output_with_retry(
                await self.llm.ainvoke(
                    self.editing_prompt.format(**input_data)
                )
            )
            return {**state, "knowledge_editing_input": result}

        # For each inferred relationship, create a knowledge editing input
        knowledge_editing_inputs = []
        for relationship in complex_relationship.relationships:
            input_data = {
                "entities": f"{state['entity_chain'][0]}, {state['entity_chain'][-1]}",
                "relationships": relationship,  # Pass the current relationship
                "descriptions": self._format_entity_descriptions(state["entity_descriptions"])
            }
            result = await self._parse_editing_output_with_retry(
                await self.llm.ainvoke(
                    self.editing_prompt.format(**input_data)
                )
            )
            knowledge_editing_inputs.append(result)

        return {**state, "knowledge_editing_input": knowledge_editing_inputs}
    
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
                
                # Clean the output string
                json_str = output.strip()
                
                # Remove any text before the first { and after the last }
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = json_str[start:end]
                else:
                    raise ValueError("No valid JSON object found in the output")
                
                # Try to parse the JSON
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    print(f"Problematic JSON string: {json_str}")
                    raise
                
                # Ensure generalization and locality are properly structured
                if 'generalization' in data:
                    if isinstance(data['generalization'], dict):
                        data['generalization'] = Generalization(**data['generalization'])
                    else:
                        data['generalization'] = Generalization(
                            generalization_prompt=data.get('generalization_prompt', ''),
                            generalization_answer=data.get('generalization_answer', '')
                        )
                
                if 'locality' in data:
                    if isinstance(data['locality'], dict):
                        data['locality'] = Locality(**data['locality'])
                    else:
                        data['locality'] = Locality(
                            locality_prompt=data.get('locality_prompt', ''),
                            locality_answer=data.get('locality_answer', '')
                        )

                if 'portability' in data:
                    if isinstance(data['portability'], dict):
                        data['portability'] = Portability(**data['portability'])
                    else:
                        data['portability'] = Portability(
                            portability_prompt=data.get('portability_prompt', ''),
                            portability_answer=data.get('portability_answer', '')
                        )
                
                # Ensure rephrase is a list
                if 'rephrase' not in data:
                    data['rephrase'] = []
                elif not isinstance(data['rephrase'], list):
                    data['rephrase'] = [data['rephrase']]
                
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
                        
                        Error message:
                        {error}
                        
                        Please provide a valid JSON response that strictly follows the schema.
                        The response must be a valid JSON object starting with {{ and ending with }}.
                        Do not include any text before or after the JSON object.
                        """
                    )
                    retry_chain = retry_prompt | self.llm
                    output = await retry_chain.ainvoke({
                        "schema": KnowledgeEditingInput.model_json_schema(),
                        "output": output,
                        "error": str(e)
                    })
                    continue
                else:
                    raise ValueError(f"Failed to parse knowledge editing input after {self.max_retries} attempts: {str(e)}")
    
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
        
        # Sample chains based on configuration
        print("Sampling entity chains...")
        if self.use_communities:
            print(f"Using community-based sampling with {self.n_communities} communities")
            if self.n_samples_per_community is None:
                print("Getting all possible chains per community")
            else:
                print(f"Getting {self.n_samples_per_community} chains per community")
            community_chains = optimized_extract_community_chains(
                self.graphml_path,
                self.communities_parquet_path,
                self.chain_length,
                self.n_communities,
                self.n_samples_per_community
            )
            # Flatten community chains into a single list while preserving community IDs
            chains = []
            chain_to_community = {}  # Map to store chain -> community_id mapping
            for community_id, community_chain_set in community_chains.items():
                for chain in community_chain_set:
                    # Convert chain entities to lowercase
                    lower_chain = tuple(entity.lower() for entity in chain)
                    chains.append(lower_chain)
                    chain_to_community[lower_chain] = community_id
            print(f"Found total of {len(chains)} chains across all communities")
        else:
            print("Using regular chain sampling")
            raw_chains = list(optimized_extract_entity_chains(
                self.graphml_path,
                self.chain_length,
                self.n_samples
            ))
            # Convert chain entities to lowercase
            chains = [tuple(entity.lower() for entity in chain) for chain in raw_chains]
            chain_to_community = {}  # Empty dict for regular sampling
            print(f"Found {len(chains)} chains")
        
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
                    knowledge_editing_input = final_state["knowledge_editing_input"]
                    if isinstance(knowledge_editing_input, list):
                        knowledge_editing_input_dumped = [k.model_dump() for k in knowledge_editing_input]
                    elif knowledge_editing_input is not None:
                        knowledge_editing_input_dumped = knowledge_editing_input.model_dump()
                    else:
                        knowledge_editing_input_dumped = None

                    # Create ComplexRelationshipResult
                    result = ComplexRelationshipResult(
                        chain=list(chain),
                        first_entity=final_state["complex_relationship"].first_entity,
                        last_entity=final_state["complex_relationship"].last_entity,
                        entity_descriptions={
                            entity: desc["description"]
                            for entity, desc in entity_descriptions.items()
                            if entity in chain
                        },
                        relationship_descriptions=[
                            f"{source} -> {target}: {rel['description']}"
                            for (source, target), rel in relationships.items()
                            if source in chain and target in chain
                        ],
                        inferred_relationships=[
                            InferredRelationship(
                                relationship=rel,
                                evidence=final_state["complex_relationship"].evidence,
                                score=float(final_state["complex_relationship"].score) if hasattr(final_state["complex_relationship"], "score") else 0.0,
                                reasoning=final_state["complex_relationship"].reasoning if hasattr(final_state["complex_relationship"], "reasoning") else "No reasoning provided"
                            )
                            for rel in final_state["complex_relationship"].relationships
                        ]
                    )
                    
                    # Convert to dictionary and add additional fields
                    result_dict = result.model_dump()
                    result_dict["evidence"] = final_state["complex_relationship"].evidence
                    result_dict["knowledge_editing_input"] = knowledge_editing_input_dumped
                    # Add community_id if available
                    result_dict["community_id"] = chain_to_community.get(tuple(chain)) if self.use_communities else None
                    
                    batch_results.append(result_dict)
                except Exception as e:
                    print(f"Error processing chain {chain}: {str(e)}")
                    # Add a default result for failed chains
                    result = ComplexRelationshipResult(
                        chain=list(chain),
                        first_entity=chain[0],
                        last_entity=chain[-1],
                        entity_descriptions={},
                        relationship_descriptions=[],
                        inferred_relationships=[
                            InferredRelationship(
                                relationship="no_relationship",
                                evidence=[],
                                score=0.0,
                                reasoning=f"Error occurred while processing chain: {str(e)}"
                            )
                        ]
                    )
                    result_dict = result.model_dump()
                    result_dict["evidence"] = []
                    result_dict["knowledge_editing_input"] = None
                    result_dict["community_id"] = None
                    batch_results.append(result_dict)
            
            results.extend(batch_results)
        
        # Save results to JSON file with dataset name, sampling type, and chain length
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sampling_type = "community" if self.use_communities else "regular"
        output_file = os.path.join(
            self.output_dir,
            f"complex_agent_result_{self.dataset_name}_{sampling_type}_chain{self.chain_length}_{timestamp}.json"
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        
        return results
