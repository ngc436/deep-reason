from deep_reason.gen_agent.sampling import optimized_extract_entity_chains, get_all_paths, map_entities_to_descriptions

def main():
    # Example usage with a graphml file
    graphml_path = "datasets/graphs/obliqa-full/output/graph.graphml" 
    parquet_path = "datasets/graphs/obliqa-full/output/entities.parquet"
    
    # Example 2: Sample 5 random chains of length 10 using optimized version
    print("\nSampling 5 random chains of length 10 using optimized version:")
    sampled_chains = optimized_extract_entity_chains(graphml_path, chain_length=10, n_samples=5)
    for chain in sampled_chains:
        print(f"Sampled chain: {' -> '.join(chain)}")

    # Map entities to their descriptions
    print("\nMapping entities to their descriptions:")
    entity_descriptions = map_entities_to_descriptions(sampled_chains, parquet_path)
    
    # Print descriptions for each entity in the chains
    for chain in sampled_chains:
        print(f"\nChain: {' -> '.join(chain)}")
        for entity in chain:
            desc = entity_descriptions[entity]
            print(f"  Entity: {entity}")
            print(f"    Description: {desc['description']}")
            print(f"    Type: {desc['type']}")
            print(f"    Frequency: {desc['frequency']}")
            print(f"    Degree: {desc['degree']}")

if __name__ == "__main__":
    main() 