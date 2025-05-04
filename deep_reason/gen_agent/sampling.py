import networkx as nx
from typing import Set, Tuple, List, Optional, Dict, Any
from itertools import combinations
import random
from collections import deque
import pandas as pd

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
    
    Args:
        graphml_path (str): Path to the .graphml file
        chain_length (int): Desired length of the entity chains
        n_samples (int): Number of chains to sample
        max_attempts (int): Maximum number of attempts to find valid chains
        
    Returns:
        Set[Tuple[str, ...]]: Set of entity chains
    """
    G = nx.read_graphml(graphml_path)
    chains = set()
    nodes = list(G.nodes())
    attempts = 0
    
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
            if not G.has_edge(chain[0], chain[-1]):
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
        matching_rows = df[df['title'] == entity]
        
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
    # Read the relationships parquet file
    df = pd.read_parquet(relationships_parquet_path)
    
    # Create a dictionary to store relationships
    relationships = {}
    
    # Process each chain
    for chain in entity_chains:
        # Get all pairs of neighboring entities in the chain
        for i in range(len(chain) - 1):
            source = chain[i]
            target = chain[i + 1]
            
            # Find matching relationships in the dataframe
            matching_relationships = df[
                (df['source'] == source) & 
                (df['target'] == target)
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
                # If no relationship found, create an empty entry
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
    
    return relationships


