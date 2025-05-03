from deep_reason.gen_agent.sampling import optimized_extract_entity_chains, get_all_paths

def main():
    # Example usage with a graphml file
    graphml_path = "datasets/graphs/obliqa-full/output/graph.graphml"  # Replace with your actual graphml file path
    
    # Example 1: Get all chains of length 3
    # print("Getting all chains of length 3:")
    # all_chains = extract_entity_chains(graphml_path, chain_length=3)
    # print(f"Found {len(all_chains)} chains:")
    # for chain in all_chains:
    #     print(f"Chain: {' -> '.join(chain)}")
    
    # Example 2: Sample 5 random chains of length 10 using optimized version
    print("\nSampling 5 random chains of length 10 using optimized version:")
    sampled_chains = optimized_extract_entity_chains(graphml_path, chain_length=10, n_samples=5)
    for chain in sampled_chains:
        print(f"Sampled chain: {' -> '.join(chain)}")
    
    # # Example 3: Get all paths of length 3
    # print("\nGetting all paths of length 3:")
    # all_paths = get_all_paths(graphml_path, chain_length=3)
    # print(f"Found {len(all_paths)} paths:")
    # for path in all_paths:
    #     print(f"Path: {' -> '.join(path)}")
    
    # # Example 4: Sample 5 random paths of length 3
    # print("\nSampling 5 random paths of length 3:")
    # sampled_paths = get_all_paths(graphml_path, chain_length=3, n_samples=5)
    # for path in sampled_paths:
    #     print(f"Sampled path: {' -> '.join(path)}")

if __name__ == "__main__":
    main() 