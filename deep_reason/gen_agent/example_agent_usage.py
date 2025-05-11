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
        n_communities=5,
        n_samples_per_community=30,
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
                print(f"    Subject: {ei['subject']}")
                print(f"    Prompt: {ei['prompt']}")
                print(f"    Target New: {ei['target_new']}")
                print(f"    Portability:")
                print(f"      Logical Generalization:")
                for lg in ei['portability']['logical_generalization']:
                    print(f"        Prompt: {lg['prompt']}")
                    print(f"        Ground Truth: {lg['ground_truth']}")
                print(f"      Reasoning:")
                for r in ei['portability']['reasoning']:
                    print(f"        Prompt: {r['prompt']}")
                    print(f"        Ground Truth: {r['ground_truth']}")
                print(f"      Subject Aliasing:")
                for sa in ei['portability']['subject_aliasing']:
                    print(f"        Prompt: {sa['prompt']}")
                    print(f"        Ground Truth: {sa['ground_truth']}")
                print(f"    Locality:")
                print(f"      Relation Specificity:")
                for rs in ei['locality']['relation_specificity']:
                    print(f"        Prompt: {rs['prompt']}")
                    print(f"        Ground Truth: {rs['ground_truth']}")
        elif editing_input is not None:
            print(f"  Subject: {editing_input['subject']}")
            print(f"  Prompt: {editing_input['prompt']}")
            print(f"  Target New: {editing_input['target_new']}")
            print(f"  Portability:")
            print(f"    Logical Generalization:")
            for lg in editing_input['portability']['logical_generalization']:
                print(f"      Prompt: {lg['prompt']}")
                print(f"      Ground Truth: {lg['ground_truth']}")
            print(f"    Reasoning:")
            for r in editing_input['portability']['reasoning']:
                print(f"      Prompt: {r['prompt']}")
                print(f"      Ground Truth: {r['ground_truth']}")
            print(f"    Subject Aliasing:")
            for sa in editing_input['portability']['subject_aliasing']:
                print(f"      Prompt: {sa['prompt']}")
                print(f"      Ground Truth: {sa['ground_truth']}")
            print(f"  Locality:")
            print(f"    Relation Specificity:")
            for rs in editing_input['locality']['relation_specificity']:
                print(f"      Prompt: {rs['prompt']}")
                print(f"      Ground Truth: {rs['ground_truth']}")
        else:
            print("  Knowledge Editing Input: None")


if __name__ == "__main__":
    asyncio.run(main()) 