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
    
    # Example 1: Regular chain sampling
    # print("\nExample 1: Regular chain sampling")
    # agent = ComplexRelationshipAgent(
    #     llm=llm,
    #     graphml_path="datasets/graphs/tat_data_3/output/graph.graphml",
    #     entities_parquet_path="datasets/graphs/tat_data_3/output/entities.parquet",
    #     relationships_parquet_path="datasets/graphs/tat_data_3/output/relationships.parquet",
    #     chain_length=3,
    #     n_samples=2,
    #     dataset_name="tat_data_3"
    # )
    
    # # Infer relationships and prepare knowledge editing inputs
    # results = await agent.infer_relationships()
    # print_results(results)
    
    # Example 2: Community-based sampling
    print("\nExample 2: Community-based sampling")
    agent = ComplexRelationshipAgent(
        llm=llm,
        graphml_path="datasets/graphs/obliqa-full/output/graph.graphml",
        entities_parquet_path="datasets/graphs/obliqa-full/output/entities.parquet",
        relationships_parquet_path="datasets/graphs/obliqa-full/output/relationships.parquet",
        chain_length=3,
        n_samples=2,  # This will be ignored when use_communities is True
        use_communities=True,
        communities_parquet_path="datasets/graphs/obliqa-full/output/communities.parquet",
        n_communities=1,
        n_samples_per_community=2,
        dataset_name="obliqa-full"
    )
    
    # Infer relationships and prepare knowledge editing inputs
    results = await agent.infer_relationships()
    print_results(results)


def print_results(results):
    """Helper function to print results in a readable format"""
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
        editing_input = result['knowledge_editing_input']
        if isinstance(editing_input, list):
            for idx, ei in enumerate(editing_input):
                print(f"  Knowledge Editing Input {idx+1}:")
                print(f"    Edit Prompt: {ei['edit_prompt']}")
                print(f"    Subject: {ei['subject']}")
                print(f"    Target: {ei['target']}")
                print(f"    Generalization:")
                print(f"      Prompt: {ei['generalization']['generalization_prompt']}")
                print(f"      Answer: {ei['generalization']['generalization_answer']}")
                print(f"    Locality:")
                print(f"      Prompt: {ei['locality']['locality_prompt']}")
                print(f"      Answer: {ei['locality']['locality_answer']}")
                print(f"    Portability:")
                print(f"      Prompt: {ei['portability']['portability_prompt']}")
                print(f"      Answer: {ei['portability']['portability_answer']}")
                print(f"    Rephrase:")
                for rephrase in ei['rephrase']:
                    print(f"      - {rephrase}")
        elif editing_input is not None:
            print(f"  Edit Prompt: {editing_input['edit_prompt']}")
            print(f"  Subject: {editing_input['subject']}")
            print(f"  Target: {editing_input['target']}")
            print(f"  Generalization:")
            print(f"    Prompt: {editing_input['generalization']['generalization_prompt']}")
            print(f"    Answer: {editing_input['generalization']['generalization_answer']}")
            print(f"  Locality:")
            print(f"    Prompt: {editing_input['locality']['locality_prompt']}")
            print(f"    Answer: {editing_input['locality']['locality_answer']}")
            print(f"  Portability:")
            print(f"    Prompt: {editing_input['portability']['portability_prompt']}")
            print(f"    Answer: {editing_input['portability']['portability_answer']}")
            print(f"  Rephrase:")
            for rephrase in editing_input['rephrase']:
                print(f"    - {rephrase}")
        else:
            print("  Knowledge Editing Input: None")


if __name__ == "__main__":
    asyncio.run(main()) 