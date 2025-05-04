from deep_reason.gen_agent.sampling import (optimized_extract_entity_chains, 
                                            map_entities_to_descriptions, 
                                            extract_chain_relationships)

def main():
    # Example usage with a graphml file
    graphml_path = "datasets/graphs/obliqa-full/output/graph.graphml" 
    parquet_path = "datasets/graphs/obliqa-full/output/entities.parquet"
    relationships_path = "datasets/graphs/obliqa-full/output/relationships.parquet"
    
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

    # Extract relationships between entities in chains
    print("\nExtracting relationships between entities:")
    relationships = extract_chain_relationships(sampled_chains, relationships_path)
    
    # Print relationships for each chain
    for chain in sampled_chains:
        print(f"\nChain: {' -> '.join(chain)}")
        for i in range(len(chain) - 1):
            source = chain[i]
            target = chain[i + 1]
            rel = relationships[(source, target)]
            print(f"  Relationship: {source} -> {target}")
            print(f"    ID: {rel['id']}")
            print(f"    Human Readable ID: {rel['human_readable_id']}")
            print(f"    Description: {rel['description']}")
            print(f"    Weight: {rel['weight']}")
            print(f"    Combined Degree: {rel['combined_degree']}")
            print(f"    Number of Text Units: {len(rel['text_unit_ids'])}")

if __name__ == "__main__":
    main() 