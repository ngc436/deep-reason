import asyncio
from deep_reason.gen_agent.agent import ComplexRelationshipAgent
from deep_reason.rag.utils import VLLMChatOpenAI


async def main():
    # Initialize VLLM client
    llm = VLLMChatOpenAI(
        model_name="/model",
        temperature=0.2,
        max_tokens=6000,
        top_p=0.95,
        base_url="http://10.32.2.11:8164/v1",
        api_key="token-abc123",
        no_think=True
    )
    
    # Initialize the agent
    agent = ComplexRelationshipAgent(
        llm=llm,
        graphml_path="datasets/graphs/tat_data_3/output/graph.graphml",
        entities_parquet_path="datasets/graphs/tat_data_3/output/entities.parquet",
        relationships_parquet_path="datasets/graphs/tat_data_3/output/relationships.parquet",
        chain_length=3,
        n_samples=10
    )
    
    # Infer relationships and prepare knowledge editing inputs
    results = await agent.infer_relationships()

    # Print results
    for i, result in enumerate(results):
        print(f"\nChain {i + 1}:")
        print(f"Chain: {' -> '.join(result['chain'])}")
        print(f"First Entity: {result['first_entity']}")
        print(f"Last Entity: {result['last_entity']}")
        print("Entity Descriptions:")
        for entity, desc in result['entity_descriptions'].items():
            print(f"  - {entity}: {desc}")
        print("Relationship Descriptions:")
        for rel in result['relationship_descriptions']:
            print(f"  - {rel}")
        print("Inferred Relationships:")
        for relationship in result['inferred_relationships']:
            print(f"  - {relationship}")
        print("Evidence:")
        for evidence in result['evidence']:
            print(f"  - {evidence}")
        
        # Print knowledge editing input if available
        if result['knowledge_editing_input']:
            print("\nKnowledge Editing Input:")
            editing_input = result['knowledge_editing_input']
            print(f"  Edit Prompt: {editing_input['edit_prompt']}")
            print(f"  Subject: {editing_input['subject']}")
            print(f"  Target: {editing_input['target']}")
            print(f"  Generalization Prompt: {editing_input['generalization_prompt']}")
            print(f"  Locality Prompt: {editing_input['locality_prompt']}")
            print(f"  Portability Prompt: {editing_input['portability_prompt']}")
        else:
            print("\nNo knowledge editing input available for this chain")

if __name__ == "__main__":
    asyncio.run(main()) 