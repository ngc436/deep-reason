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
from deep_reason.gen_agent.prompts import COMPLEX_RELATIONSHIPS_PROMPT, PREPARE_FOR_KNOWLEDGE_EDITING_PROMPT_WIKIDATA_RECENT_TYPE
from deep_reason.gen_agent.scheme import (
    ComplexRelationship,
    ComplexRelationshipResult,
    WikidataRecentKnowledgeEditingInput,
    InferredRelationship,
    Locality,
    Portability
)


class AgentState(TypedDict):
    """State for the ComplexRelationshipAgent workflow"""
    entity_chain: List[str]
    entity_descriptions: Dict[str, Dict[str, Any]]
    relationships: Dict[Tuple[str, str], Dict[str, Any]]
    complex_relationship: Optional[ComplexRelationship]
    knowledge_editing_input: Optional[WikidataRecentKnowledgeEditingInput]
    complex_relationship_result: Optional[ComplexRelationshipResult]

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
        selected_community_ids: Optional[List[int]] = None,
        min_entities_per_community: Optional[int] = None,
        max_entities_per_community: Optional[int] = None,
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
        self.selected_community_ids = selected_community_ids
        self.min_entities_per_community = min_entities_per_community
        self.max_entities_per_community = max_entities_per_community
        self.dataset_name = dataset_name
        
        # Validate community-related parameters
        if self.use_communities:
            if not self.communities_parquet_path:
                raise ValueError("communities_parquet_path must be provided when use_communities is True")
            if not self.selected_community_ids and not self.n_communities:
                raise ValueError("Either selected_community_ids or n_communities must be provided when use_communities is True")
            # n_samples_per_community can be None to get all possible chains
            if self.min_entities_per_community is not None and self.max_entities_per_community is not None:
                if self.min_entities_per_community > self.max_entities_per_community:
                    raise ValueError("min_entities_per_community cannot be greater than max_entities_per_community")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for relationship inference and knowledge editing input preparation"""
        
        # Create the output parser for complex relationships
        relationship_parser = PydanticOutputParser(pydantic_object=ComplexRelationship)
        
        # Create the output parser for knowledge editing input
        editing_parser = PydanticOutputParser(pydantic_object=WikidataRecentKnowledgeEditingInput)
        
        # Create the prompt templates
        self.relationship_prompt = PromptTemplate(
            input_variables=["entity_chain", "entity_descriptions", "relationships"],
            template=COMPLEX_RELATIONSHIPS_PROMPT,
            partial_variables={"schema": relationship_parser.get_format_instructions()}
        )
        
        self.editing_prompt = PromptTemplate(
            input_variables=["entities", "relationships", "descriptions"],
            template=PREPARE_FOR_KNOWLEDGE_EDITING_PROMPT_WIKIDATA_RECENT_TYPE,
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
        empty_count = 0
        total_count = 0
        
        for (source, target), rel in relationships.items():
            total_count += 1
            if not rel['description']:
                empty_count += 1
                print(f"Warning: Empty relationship description for {source} -> {target}")
                # Try to find a default description based on the relationship type
                if 'human_readable_id' in rel and rel['human_readable_id']:
                    formatted.append(f"{source} -> {target}: {rel['human_readable_id']}")
                else:
                    formatted.append(f"{source} -> {target}: connected")
            else:
                formatted.append(f"{source} -> {target}: {rel['description']}")
        
        if empty_count > 0:
            print(f"Found {empty_count} empty relationships out of {total_count} total relationships")
        
        return "\n".join(formatted)
    
    def _format_entity_chain(self, chain: List[str]) -> str:
        """Format entity chain with arrows between entities
        
        Args:
            chain: List of entities in the chain
            
        Returns:
            Formatted string with entities joined by arrows
        """
        return " -> ".join(chain)
    
    async def _parse_editing_output_with_retry(self, output: Any) -> WikidataRecentKnowledgeEditingInput:
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
                
                # Ensure portability and locality are properly structured
                if 'portability' in data:
                    if isinstance(data['portability'], dict):
                        data['portability'] = Portability(**data['portability'])
                    else:
                        data['portability'] = Portability(
                            logical_generalization=data.get('logical_generalization', []),
                            reasoning=data.get('reasoning', []),
                            subject_aliasing=data.get('subject_aliasing', [])
                        )
                
                if 'locality' in data:
                    if isinstance(data['locality'], dict):
                        data['locality'] = Locality(**data['locality'])
                    else:
                        data['locality'] = Locality(
                            relation_specificity=data.get('relation_specificity', [])
                        )
                
                return WikidataRecentKnowledgeEditingInput(**data)
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
                        "schema": WikidataRecentKnowledgeEditingInput.model_json_schema(),
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
    
    def _create_relationship_prompt(self, chain: Tuple[str, ...], entity_descriptions: Dict[str, str], relationships: List[str]) -> str:
        """Create a prompt for relationship inference.
        
        Args:
            chain: Tuple of entities in the chain
            entity_descriptions: Dictionary mapping entities to their descriptions
            relationships: List of relationship descriptions between consecutive entities
            
        Returns:
            Formatted prompt string
        """
        # Format the chain
        chain_str = " -> ".join(chain)
        
        # Format entity descriptions
        desc_str = "\n".join(f"{entity}: {desc}" for entity, desc in entity_descriptions.items())
        
        # Format relationships
        rel_str = "\n".join(relationships)
        
        # Create the prompt
        prompt = f"""Given the following chain of entities and their relationships:

Chain: {chain_str}

Entity Descriptions:
{desc_str}

Direct Relationships:
{rel_str}

Please infer the complex relationship between the first and last entities in the chain.
Consider the intermediate entities and their relationships to understand the full context.

Provide your response in the following JSON format:
{{
    "first_entity": "name of first entity",
    "last_entity": "name of last entity",
    "relationships": ["list of inferred relationships"],
    "evidence": ["list of evidence supporting the relationships"],
    "score": 0.0,  # confidence score between 0 and 1
    "reasoning": "explanation of how you arrived at these relationships"
}}

The response must be a valid JSON object starting with {{ and ending with }}.
Do not include any text before or after the JSON object.
"""
        return prompt

    def _parse_llm_response(self, response: Any, chain: Tuple[str, ...], entity_descriptions: Dict[str, str], relationships: List[str]) -> Dict[str, Any]:
        """Parse the LLM response into a result dictionary.
        
        Args:
            response: LLM response
            chain: Original chain of entities
            entity_descriptions: Dictionary mapping entities to their descriptions
            relationships: List of relationship descriptions
            
        Returns:
            Dictionary containing the parsed result
        """
        try:
            # Extract content from AIMessage if needed
            if hasattr(response, 'content'):
                response = response.content
            
            # Parse the JSON response
            data = json.loads(response.strip())
            
            # Create the result dictionary
            result = {
                "chain": list(chain),
                "first_entity": data.get("first_entity", chain[0]),
                "last_entity": data.get("last_entity", chain[-1]),
                "entity_descriptions": entity_descriptions,
                "relationship_descriptions": relationships,
                "inferred_relationships": data.get("relationships", []),
                "evidence": data.get("evidence", []),
                "score": float(data.get("score", 0.0)),
                "reasoning": data.get("reasoning", "No reasoning provided")
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Return a default result if parsing fails
            return {
                "chain": list(chain),
                "first_entity": chain[0],
                "last_entity": chain[-1],
                "entity_descriptions": entity_descriptions,
                "relationship_descriptions": relationships,
                "inferred_relationships": ["no_relationship"],
                "evidence": [],
                "score": 0.0,
                "reasoning": f"Error parsing response: {str(e)}"
            }

    async def infer_relationships(self) -> List[Dict[str, Any]]:
        """
        Infer relationships between entities in chains using the LLM.
        Returns a list of dictionaries containing the chain, entity descriptions, relationship descriptions,
        and inferred relationships.
        """
        print("Sampling entity chains...")
        
        if self.use_communities:
            print("Using community-based sampling")
            if self.selected_community_ids:
                print(f"Using {len(self.selected_community_ids)} selected communities")
            else:
                print(f"Using {self.n_communities} random communities")
            print(f"Getting {self.n_samples_per_community} chains per community")
            print(f"Minimum entities per community: {self.min_entities_per_community}")
            print(f"Maximum entities per community: {self.max_entities_per_community}")
            
            # Get chains from communities
            community_chains = optimized_extract_community_chains(
                self.graphml_path,
                self.communities_parquet_path,
                self.chain_length,
                self.n_communities,
                self.n_samples_per_community,
                self.selected_community_ids,
                self.min_entities_per_community,
                self.max_entities_per_community
            )
            
            # Flatten chains from all communities
            all_chains = set()
            for community_id, chains in community_chains.items():
                all_chains.update(chains)
            print(f"Found total of {len(all_chains)} valid chains across all communities")
            
            if not all_chains:
                print("Warning: No valid chains found in any community")
                return []
                
            chains = all_chains
        else:
            print("Using random sampling")
            chains = optimized_extract_entity_chains(
                self.graphml_path,
                self.chain_length,
                self.n_samples
            )
            print(f"Found {len(chains)} valid chains")
        
        if not chains:
            print("No valid chains found")
            return []
        
        print("Mapping entities to descriptions...")
        entity_descriptions = map_entities_to_descriptions(chains, self.entities_parquet_path)
        
        print("Extracting chain relationships...")
        relationships = extract_chain_relationships(chains, self.relationships_parquet_path)
        
        # Process chains in batches
        results = []
        batch_size = 5  # Process 5 chains at a time
        
        for i in range(0, len(chains), batch_size):
            batch_chains = list(chains)[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} of {(len(chains) + batch_size - 1)//batch_size}")
            
            for chain in batch_chains:
                try:
                    # Convert chain entities to lowercase and ensure they are ASCII
                    try:
                        chain = tuple(entity.encode('ascii', 'ignore').decode('ascii').lower() for entity in chain)
                    except Exception as e:
                        print(f"Warning: Skipping chain with encoding issues: {e}")
                        continue
                        
                    if not all(chain):  # Skip if any entity is empty after conversion
                        print("Warning: Skipping chain with empty entities after conversion")
                        continue
                    
                    # Get entity descriptions for this chain
                    chain_descriptions = {
                        entity: entity_descriptions.get(entity, {}).get('description', '')
                        for entity in chain
                    }
                    
                    # Get relationship descriptions for this chain
                    chain_relationships = []
                    for j in range(len(chain) - 1):
                        source, target = chain[j], chain[j + 1]
                        rel = relationships.get((source, target), {})
                        if rel and rel.get('description'):
                            chain_relationships.append(f"{source} -> {target}: {rel['description']}")
                    
                    # Prepare the prompt for relationship inference
                    prompt = self._create_relationship_prompt(chain, chain_descriptions, chain_relationships)
                    
                    # Get LLM response
                    response = await self.llm.ainvoke(prompt)
                    
                    # Parse the response
                    result = self._parse_llm_response(response, chain, chain_descriptions, chain_relationships)
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing chain: {e}")
                    continue
        
        return results
