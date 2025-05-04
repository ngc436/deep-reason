from typing import Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
import json

from deep_reason.gen_agent.sampling import (
    optimized_extract_entity_chains,
    map_entities_to_descriptions,
    extract_chain_relationships
)
from deep_reason.gen_agent.prompts import COMPLEX_RELATIONSHIPS_PROMPT


class ComplexRelationship(BaseModel):
    """Model for complex relationship between chain endpoints"""
    first_entity: str = Field(description="First entity in the chain")
    last_entity: str = Field(description="Last entity in the chain")
    evidence: List[str] = Field(description="List of evidence to infer relationship between first and last entities in the chain based on the relationships and descriptions.")
    relationships: List[str] = Field(description="Inferred relationships between first and last entities in a form of concise sentences. Each of the two entities should be in the sentence.")


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
        max_retries: int = 3
    ):
        self.llm = llm
        self.graphml_path = graphml_path
        self.entities_parquet_path = entities_parquet_path
        self.relationships_parquet_path = relationships_parquet_path
        self.chain_length = chain_length
        self.n_samples = n_samples
        self.max_retries = max_retries
        
        # Initialize the chain
        self.chain = self._build_chain()
    
    def _build_chain(self) -> Runnable:
        """Build the LangChain pipeline for relationship inference"""
        
        # Create the output parser
        parser = PydanticOutputParser(pydantic_object=ComplexRelationship)
        
        # Create the prompt template with Pydantic schema instructions
        prompt = PromptTemplate(
            input_variables=["entity_chain", "entity_descriptions", "relationships"],
            template=COMPLEX_RELATIONSHIPS_PROMPT,
            partial_variables={"schema": parser.get_format_instructions()}
        )
        
        # Build the chain
        return (
            RunnableParallel(
                entity_chain=lambda x: self._format_entity_chain(x["entity_chain"]),
                entity_descriptions=lambda x: self._format_entity_descriptions(x["entity_descriptions"]),
                relationships=lambda x: self._format_relationships(x["relationships"]),
                # schema=lambda _: ComplexRelationship.model_json_schema()
            )
            | prompt
            | self.llm
            | RunnableLambda(self._parse_output_with_retry)
        )
    
    async def _parse_output_with_retry(self, output: Any) -> ComplexRelationship:
        """Parse the LLM output with retry logic"""
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
                        relationship="no_relationship",
                        evidence=[]
                    )
    
    def _format_entity_descriptions(self, descriptions: Dict[str, Dict[str, Any]]) -> str:
        """Format entity descriptions for the prompt"""
        formatted = []
        for entity, desc in descriptions.items():
            formatted.append(f"{entity}: {desc['description']}")
            # formatted.append(f"Entity: {entity}")
            # formatted.append(f"  Description: {desc['description']}")
            # formatted.append(f"  Type: {desc['type']}")
            # formatted.append(f"  Frequency: {desc['frequency']}")
            # formatted.append(f"  Degree: {desc['degree']}")
        return "\n".join(formatted)
    
    def _format_relationships(self, relationships: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
        """Format relationships for the prompt"""
        formatted = []
        for (source, target), rel in relationships.items():
            formatted.append(f"{source} -> {target}: {rel['description']}")
            # formatted.append(f"Relationship: {source} -> {target}")
            # formatted.append(f"  Description: {rel['description']}")
            # formatted.append(f"  Weight: {rel['weight']}")
            # formatted.append(f"  Combined Degree: {rel['combined_degree']}")
        return "\n".join(formatted)
    
    def _format_entity_chain(self, chain: List[str]) -> str:
        """Format entity chain with arrows between entities
        
        Args:
            chain: List of entities in the chain
            
        Returns:
            Formatted string with entities joined by arrows
        """
        return " -> ".join(chain)
    
    async def infer_relationships(self) -> List[Dict[str, Any]]:
        """Infer complex relationships for sampled chains"""
        
        # Sample chains
        chains = optimized_extract_entity_chains(
            self.graphml_path,
            self.chain_length,
            self.n_samples
        )
        
        # Get entity descriptions
        entity_descriptions = map_entities_to_descriptions(
            chains,
            self.entities_parquet_path
        )
        
        # Get relationships
        relationships = extract_chain_relationships(
            chains,
            self.relationships_parquet_path
        )
        
        # Process each chain
        results = []
        for chain in chains:
            try:
                # Prepare input for the chain
                input_data = {
                    "entity_chain": chain,
                    "entity_descriptions": entity_descriptions,
                    "relationships": relationships
                }
                
                # Run the chain
                result = await self.chain.ainvoke(input_data)
                
                # Convert to dictionary
                result_dict = result.model_dump()
                result_dict["chain"] = list(chain)
                
                results.append(result_dict)
            except Exception as e:
                print(f"Error processing chain {chain}: {str(e)}")
                # Add a default result for failed chains
                results.append({
                    "chain": list(chain),
                    "first_entity": chain[0],
                    "last_entity": chain[-1],
                    "relationship": "no_relationship",
                    "evidence": []
                })
        
        return results
