from sampling import optimized_extract_community_chains
import pandas as pd
import networkx as nx

def main():
    # First, let's examine the input data
    print("Reading input files...")
    
    # Read and examine the communities file
    communities_df = pd.read_parquet("./datasets/graphs/tat_data_3/output/communities.parquet")
    print("\nCommunities DataFrame info:")
    print(f"Number of communities: {len(communities_df)}")
    
    # Get sample of entity IDs from first community
    sample_entity_ids = communities_df['entity_ids'].iloc[0][:5]
    print("\nSample entity IDs from first community:")
    print(sample_entity_ids)
    
    # Read and examine the graph
    G = nx.read_graphml("./datasets/graphs/tat_data_3/output/graph.graphml")
    print("\nGraph info:")
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of edges: {len(G.edges())}")
    
    # Show sample of node names
    print("\nSample node names from graph:")
    print(list(G.nodes())[:5])
    
    # Example usage of optimized_extract_community_chains
    print("\nExtracting community chains...")
    community_chains = optimized_extract_community_chains(
        graphml_path="./datasets/graphs/tat_data_3/output/graph.graphml",
        communities_parquet_path="./datasets/graphs/tat_data_3/output/communities.parquet",
        chain_length=4,  # Use chain length of 2
        n_communities=1,  # Sample from 1 community
        n_samples_per_community=None,  # Get 3 chains per community
        max_attempts=500  # Maximum attempts to find valid chains
    )
    
    # Print the results
    print("\nExtracted community chains:")
    if not community_chains:
        print("No chains were extracted. This might be due to:")
        print("1. Communities being too small for the requested chain length")
        print("2. Communities not having enough connected nodes")
        print("3. Not enough attempts to find valid chains")
    else:
        for community_id, chains in community_chains.items():
            print(f"\nCommunity {community_id}:")
            for chain in chains:
                print(f"  Chain (UUIDs): {' -> '.join(chain)}")
    
    # # Clean up sample files
    # os.remove("sample_graph.graphml")
    # os.remove("sample_communities.parquet")

if __name__ == "__main__":
    main() 