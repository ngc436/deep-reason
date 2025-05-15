import networkx as nx
from typing import Set, Tuple, List, Optional, Dict, Any
from itertools import combinations
import random
from collections import deque
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_entity_chains(graphml_path: str, chain_length: int, n_samples: Optional[int] = None) -> Set[Tuple[str, ...]]:
    """
    Extract chains of entities of specified length from a .graphml file.
    Optionally sample a specific number of chains.
    
    Args:
        graphml_path (str): Path to the .graphml file
        chain_length (int): Desired length of the entity chains
        n_samples (Optional[int]): Number of chains to sample. If None, returns all chains.
        
    Returns:
        Set[Tuple[str, ...]]: Set of entity chains, where each chain is a tuple of entity names
    """
    # Read the graph from the .graphml file
    G = nx.read_graphml(graphml_path)
    
    # Get all nodes (entities) from the graph
    entities = list(G.nodes())
    
    # Initialize set to store valid chains
    chains = set()
    
    # If n_samples is specified, use random sampling
    if n_samples is not None:
        # Use BFS to find chains more efficiently
        for _ in range(n_samples):
            # Start from a random node
            start_node = random.choice(entities)
            chain = [start_node]
            
            # Perform BFS up to the desired length
            queue = deque([(start_node, chain)])
            while queue and len(chains) < n_samples:
                current_node, current_chain = queue.popleft()
                
                if len(current_chain) == chain_length:
                    chains.add(tuple(current_chain))
                    continue
                
                # Get neighbors and shuffle them for randomness
                neighbors = list(G.neighbors(current_node))
                random.shuffle(neighbors)
                
                for neighbor in neighbors:
                    if neighbor not in current_chain:
                        new_chain = current_chain + [neighbor]
                        queue.append((neighbor, new_chain))
            
            if len(chains) >= n_samples:
                break
    else:
        # Original implementation for getting all chains
        for chain in combinations(entities, chain_length):
            is_valid_chain = True
            for i in range(len(chain) - 1):
                if not G.has_edge(chain[i], chain[i + 1]):
                    is_valid_chain = False
                    break
            
            if is_valid_chain:
                chains.add(chain)
    
    return chains

def get_all_paths(graphml_path: str, chain_length: int, n_samples: Optional[int] = None) -> List[List[str]]:
    """
    Find paths of specified length in the graph, optionally sampling a specific number.
    
    Args:
        graphml_path (str): Path to the .graphml file
        chain_length (int): Desired length of the entity chains
        n_samples (Optional[int]): Number of paths to sample. If None, returns all paths.
        
    Returns:
        List[List[str]]: List of entity chains, where each chain is a list of entity names
    """
    G = nx.read_graphml(graphml_path)
    all_paths = []
    
    # If n_samples is specified, use random sampling
    if n_samples is not None:
        nodes = list(G.nodes())
        while len(all_paths) < n_samples:
            # Randomly select start and end nodes
            start_node = random.choice(nodes)
            end_node = random.choice(nodes)
            
            if start_node != end_node:
                try:
                    # Get a random path between start and end nodes
                    path = nx.shortest_path(G, start_node, end_node)
                    if len(path) == chain_length:
                        all_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
    else:
        # Original implementation for getting all paths
        for start_node in G.nodes():
            for end_node in G.nodes():
                if start_node != end_node:
                    try:
                        paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=chain_length-1))
                        valid_paths = [path for path in paths if len(path) == chain_length]
                        all_paths.extend(valid_paths)
                    except nx.NetworkXNoPath:
                        continue
    
    return all_paths[:n_samples] if n_samples is not None else all_paths

def optimized_extract_entity_chains(graphml_path: str, chain_length: int, n_samples: int, max_attempts: int = 1000) -> Set[Tuple[str, ...]]:
    """
    Extract chains of entities using an optimized random walk strategy.
    This is more efficient for long chains than the BFS approach.
    Ensures there is no direct connection between first and last elements in the chain.
    Also ensures all chains are unique by checking both forward and reverse order.
    
    Args:
        graphml_path (str): Path to the .graphml file
        chain_length (int): Desired length of the entity chains
        n_samples (int): Number of chains to sample
        max_attempts (int): Maximum number of attempts to find valid chains
        
    Returns:
        Set[Tuple[str, ...]]: Set of unique entity chains
    """
    G = nx.read_graphml(graphml_path)
    chains = set()
    nodes = list(G.nodes())
    attempts = 0
    
    def is_chain_unique(chain: List[str]) -> bool:
        """Check if a chain is unique by comparing both forward and reverse order."""
        chain_tuple = tuple(chain)
        reverse_chain_tuple = tuple(reversed(chain))
        return chain_tuple not in chains and reverse_chain_tuple not in chains
    
    while len(chains) < n_samples and attempts < max_attempts:
        attempts += 1
        # Start from a random node
        current_node = random.choice(nodes)
        chain = [current_node]
        
        # Perform random walk
        while len(chain) < chain_length:
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
                
            # Choose next node randomly
            next_node = random.choice(neighbors)
            if next_node not in chain:
                chain.append(next_node)
                current_node = next_node
            else:
                # If we hit a cycle, restart from a random node
                break
        
        # If we found a valid chain of the right length
        if len(chain) == chain_length:
            # Check if there is no direct connection between first and last elements
            if not G.has_edge(chain[0], chain[-1]) and is_chain_unique(chain):
                chains.add(tuple(chain))
    
    return chains

def map_entities_to_descriptions(entity_chains: Set[Tuple[str, ...]], parquet_path: str) -> Dict[str, Dict[str, str]]:
    """
    Map entities from chains to their descriptions from a parquet file.
    
    Args:
        entity_chains (Set[Tuple[str, ...]]): Set of entity chains from optimized_extract_entity_chains
        parquet_path (str): Path to the parquet file containing entity descriptions
        
    Returns:
        Dict[str, Dict[str, str]]: Dictionary mapping each entity to its description and other fields
    """
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    # Create a dictionary to store entity descriptions
    entity_descriptions = {}
    
    # Get all unique entities from the chains
    unique_entities = set()
    for chain in entity_chains:
        unique_entities.update(chain)
    
    # Map each entity to its description
    for entity in unique_entities:
        # Find matching row in the dataframe
        matching_rows = df[df['title'].str.lower() == entity.lower()]
        
        if not matching_rows.empty:
            # Get the first matching row
            row = matching_rows.iloc[0]
            # Create a dictionary with all relevant fields
            entity_descriptions[entity] = {
                'title': row['title'],
                'description': row['description'],
                'type': row.get('type', ''),
                'frequency': row.get('frequency', 0),
                'degree': row.get('degree', 0)
            }
        else:
            # If no matching entity found, create an entry with empty values
            entity_descriptions[entity] = {
                'title': entity,
                'description': '',
                'type': '',
                'frequency': 0,
                'degree': 0
            }
    
    return entity_descriptions

def extract_chain_relationships(entity_chains: Set[Tuple[str, ...]], relationships_parquet_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Extract relationships between neighboring entities in chains from a parquet file.
    
    Args:
        entity_chains (Set[Tuple[str, ...]]): Set of entity chains
        relationships_parquet_path (str): Path to the parquet file containing relationships
        
    Returns:
        Dict[Tuple[str, str], Dict[str, Any]]: Dictionary mapping pairs of neighboring entities to their relationship details
    """
    try:
        # Read the relationships parquet file
        print(f"Reading relationships from {relationships_parquet_path}")
        df = pd.read_parquet(relationships_parquet_path)
        print(f"Loaded {len(df)} relationships")
        
        # Print column names and first few rows for debugging
        print("Columns in relationships file:", df.columns.tolist())
        print("\nFirst few rows of relationships:")
        print(df.head())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("\nMissing values per column:")
        print(missing_values)
        
        # Convert source and target columns to lowercase for case-insensitive matching
        df['source_lower'] = df['source'].str.lower()
        df['target_lower'] = df['target'].str.lower()
        
        # Create a dictionary to store relationships
        relationships = {}
        
        # Process each chain
        for chain in entity_chains:
            # Get all pairs of neighboring entities in the chain
            for i in range(len(chain) - 1):
                source = chain[i]
                target = chain[i + 1]
                
                if not source or not target:
                    print(f"Warning: Invalid chain segment - source: '{source}', target: '{target}'")
                    continue
                
                # Find matching relationships in the dataframe using case-insensitive matching
                matching_relationships = df[
                    (df['source_lower'] == source.lower()) & 
                    (df['target_lower'] == target.lower())
                ]
                
                if not matching_relationships.empty:
                    # Get the first matching relationship
                    relationship = matching_relationships.iloc[0]
                    # Store the relationship with all available fields
                    relationships[(source, target)] = {
                        'id': relationship['id'],
                        'human_readable_id': relationship['human_readable_id'],
                        'source': relationship['source'],
                        'target': relationship['target'],
                        'description': relationship['description'],
                        'weight': relationship['weight'],
                        'combined_degree': relationship['combined_degree'],
                        'text_unit_ids': relationship['text_unit_ids']
                    }
                else:
                    # Try reverse direction
                    reverse_matching = df[
                        (df['source_lower'] == target.lower()) & 
                        (df['target_lower'] == source.lower())
                    ]
                    
                    if not reverse_matching.empty:
                        relationship = reverse_matching.iloc[0]
                        relationships[(source, target)] = {
                            'id': relationship['id'],
                            'human_readable_id': relationship['human_readable_id'],
                            'source': relationship['target'],  # Swap source and target
                            'target': relationship['source'],
                            'description': relationship['description'],
                            'weight': relationship['weight'],
                            'combined_degree': relationship['combined_degree'],
                            'text_unit_ids': relationship['text_unit_ids']
                        }
                    else:
                        # If no relationship found, create an entry with empty values
                        relationships[(source, target)] = {
                            'id': '',
                            'human_readable_id': '',
                            'source': source,
                            'target': target,
                            'description': '',
                            'weight': 0,
                            'combined_degree': 0,
                            'text_unit_ids': []
                        }
                        print(f"Warning: No relationship found for {source} -> {target}")
        
        # Print some statistics
        total_relationships = len(relationships)
        empty_relationships = sum(1 for rel in relationships.values() if not rel['description'])
        print(f"\nRelationship Statistics:")
        print(f"Total relationships: {total_relationships}")
        print(f"Empty relationships: {empty_relationships}")
        if total_relationships > 0:
            print(f"Percentage empty: {(empty_relationships/total_relationships)*100:.2f}%")
        else:
            print("No relationships found to calculate statistics")
        
        return relationships
        
    except Exception as e:
        print(f"Error reading relationships file: {str(e)}")
        raise

def optimized_extract_community_chains(
    graphml_path: str,
    communities_parquet_path: str,
    chain_length: int,
    n_communities: Optional[int] = None,
    n_samples_per_community: Optional[int] = None,
    selected_community_ids: Optional[List[int]] = None,
    min_entities_per_community: Optional[int] = None,
    max_entities_per_community: Optional[int] = None,
    max_attempts: int = 1000
) -> Dict[int, Set[Tuple[str, ...]]]:
    """
    Extract chains of entities from specific communities in a graph using an optimized random walk strategy.
    Each chain is guaranteed to be within a single community.
    
    Args:
        graphml_path: Path to the graphml file
        communities_parquet_path: Path to the communities parquet file
        chain_length: Length of chains to extract
        n_communities: Number of communities to sample from (if not using selected_community_ids)
        n_samples_per_community: Number of chains to sample per community
        selected_community_ids: List of specific community IDs to use
        min_entities_per_community: Minimum number of entities required in a community
        max_entities_per_community: Maximum number of entities allowed in a community
        max_attempts: Maximum number of attempts to find valid chains
    """
    print("Starting chain extraction process...")
    
    # Read the full graph
    G = nx.read_graphml(graphml_path)
    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Read communities from parquet file
    communities_df = pd.read_parquet(communities_parquet_path)
    print(f"Loaded {len(communities_df)} communities")
    
    # Read entities from parquet file
    entities_path = os.path.join(os.path.dirname(graphml_path), "entities.parquet")
    entities_df = pd.read_parquet(entities_path)
    print(f"Loaded {len(entities_df)} entities")
    
    # Create a mapping between UUIDs and human-readable names
    uuid_to_name = {}
    name_to_uuid = {}
    
    # Create mapping from entity IDs to titles
    for _, row in entities_df.iterrows():
        if 'id' in row and 'title' in row:
            uuid_to_name[row['id']] = row['title']
            name_to_uuid[row['title']] = row['id']
    
    print(f"Created mapping for {len(uuid_to_name)} entities")
    
    # Filter communities based on size if min/max parameters are provided
    if min_entities_per_community is not None or max_entities_per_community is not None:
        communities_df['entity_count'] = communities_df['entity_ids'].apply(len)
        if min_entities_per_community is not None:
            communities_df = communities_df[communities_df['entity_count'] >= min_entities_per_community]
        if max_entities_per_community is not None:
            communities_df = communities_df[communities_df['entity_count'] <= max_entities_per_community]
        print(f"Filtered to {len(communities_df)} communities based on size constraints")
    
    # Dictionary to store chains for each community
    community_chains = {}
    
    if selected_community_ids is not None:
        # Filter communities to only include selected IDs
        selected_communities = communities_df.loc[selected_community_ids]
        print(f"Selected {len(selected_communities)} specific communities")
        communities_to_process = selected_communities
    else:
        # Keep sampling until we get the requested number of communities with valid chains
        communities_to_process = pd.DataFrame()
        attempts = 0
        max_sampling_attempts = 100  # Maximum number of attempts to find valid communities
        
        while len(community_chains) < n_communities and attempts < max_sampling_attempts:
            attempts += 1
            # Sample more communities than needed to account for invalid ones
            sample_size = min(n_communities * 2, len(communities_df))
            sampled_communities = communities_df.sample(n=sample_size)
            
            for _, community_row in sampled_communities.iterrows():
                if len(community_chains) >= n_communities:
                    break
                    
                community_id = community_row.name
                if community_id in community_chains:
                    continue  # Skip if we already processed this community
                    
                entity_ids = community_row['entity_ids']
                print(f"\nProcessing community {len(community_chains) + 1}/{n_communities} (ID: {community_id})")
                print(f"Community has {len(entity_ids)} entities")
                
                # Convert UUIDs to human-readable names for the subgraph
                entity_names = [uuid_to_name.get(eid) for eid in entity_ids]
                entity_names = [name for name in entity_names if name is not None]
                print(f"Found {len(entity_names)} mapped names for this community")
                
                if not entity_names:
                    print("No valid names found for this community, skipping...")
                    continue
                
                # Create subgraph for this community using human-readable names
                community_subgraph = G.subgraph(entity_names)
                print(f"Created subgraph with {len(community_subgraph.nodes())} nodes and {len(community_subgraph.edges())} edges")
                
                # Skip if subgraph is too small
                if len(community_subgraph) < chain_length:
                    print(f"Subgraph too small ({len(community_subgraph)} < {chain_length}), skipping...")
                    continue
                    
                chains = set()
                
                if n_samples_per_community is None:
                    # Use all_simple_paths to find all possible chains
                    for start_node in community_subgraph.nodes():
                        for end_node in community_subgraph.nodes():
                            if start_node != end_node:
                                try:
                                    paths = list(nx.all_simple_paths(community_subgraph, start_node, end_node, cutoff=chain_length-1))
                                    valid_paths = [path for path in paths if len(path) == chain_length]
                                    for path in valid_paths:
                                        # Validate path before adding
                                        if all(node and isinstance(node, str) for node in path):
                                            chains.add(tuple(path))
                                        else:
                                            print(f"Warning: Invalid path found: {path}")
                                except nx.NetworkXNoPath:
                                    continue
                else:
                    attempts = 0
                    
                    def is_chain_unique(chain: List[str]) -> bool:
                        """Check if a chain is unique by comparing both forward and reverse order."""
                        if not all(node and isinstance(node, str) for node in chain):
                            return False
                        chain_tuple = tuple(chain)
                        reverse_chain_tuple = tuple(reversed(chain))
                        return chain_tuple not in chains and reverse_chain_tuple not in chains
                    
                    while len(chains) < n_samples_per_community and attempts < max_attempts:
                        attempts += 1
                        # Start from a random node in the community
                        current_node = random.choice(list(community_subgraph.nodes()))
                        chain = [current_node]
                        
                        # Perform random walk within the community
                        while len(chain) < chain_length:
                            neighbors = list(community_subgraph.neighbors(current_node))
                            if not neighbors:
                                break
                                
                            # Choose next node randomly
                            next_node = random.choice(neighbors)
                            if next_node not in chain:
                                chain.append(next_node)
                                current_node = next_node
                            else:
                                # If we hit a cycle, restart from a random node
                                break
                        
                        # If we found a valid chain of the right length
                        if len(chain) == chain_length and is_chain_unique(chain):
                            # Validate chain before adding
                            if all(node and isinstance(node, str) for node in chain):
                                chains.add(tuple(chain))
                            else:
                                print(f"Warning: Invalid chain found: {chain}")
                
                print(f"Found {len(chains)} valid chains")
                if chains:
                    # If n_samples_per_community is specified, limit the number of chains
                    if n_samples_per_community is not None and len(chains) > n_samples_per_community:
                        chains = set(list(chains)[:n_samples_per_community])
                        print(f"Limited to {n_samples_per_community} chains per community")
                    community_chains[community_id] = chains
        
        if len(community_chains) < n_communities:
            print(f"Warning: Could only find {len(community_chains)} valid communities out of {n_communities} requested")
    
    print(f"\nExtracted chains for {len(community_chains)} communities")
    return community_chains


