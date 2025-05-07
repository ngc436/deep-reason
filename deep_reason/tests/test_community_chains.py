import unittest
import os
import pandas as pd
import networkx as nx
from deep_reason.gen_agent.sampling import optimized_extract_community_chains

class TestCommunityChains(unittest.TestCase):
    def setUp(self):
        """Set up test data before each test method."""
        # Create a test directory if it doesn't exist
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a sample graph
        self.graph_path = os.path.join(self.test_dir, "test_graph.graphml")
        self.create_test_graph()
        
        # Create a sample communities parquet file
        self.communities_path = os.path.join(self.test_dir, "test_communities.parquet")
        self.create_test_communities()
    
    def create_test_graph(self):
        """Create a test graph with multiple communities."""
        G = nx.Graph()
        
        # Community 1: A-B-C-D
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        
        # Community 2: E-F-G-H
        G.add_edge("E", "F")
        G.add_edge("F", "G")
        G.add_edge("G", "H")
        
        # Community 3: I-J-K-L
        G.add_edge("I", "J")
        G.add_edge("J", "K")
        G.add_edge("K", "L")
        
        # Save the graph
        nx.write_graphml(G, self.graph_path)
    
    def create_test_communities(self):
        """Create a test communities parquet file."""
        communities_data = {
            'entity_ids': [
                ['A', 'B', 'C', 'D'],
                ['E', 'F', 'G', 'H'],
                ['I', 'J', 'K', 'L']
            ]
        }
        df = pd.DataFrame(communities_data)
        df.to_parquet(self.communities_path)
    
    def test_basic_functionality(self):
        """Test basic functionality of community chain extraction."""
        chains = optimized_extract_community_chains(
            graphml_path=self.graph_path,
            communities_parquet_path=self.communities_path,
            chain_length=3,
            n_communities=2,
            n_samples_per_community=2
        )
        
        # Check if we got the expected number of communities
        self.assertEqual(len(chains), 2)
        
        # Check if each community has the expected number of chains
        for community_chains in chains.values():
            self.assertLessEqual(len(community_chains), 2)
            
            # Check if each chain has the correct length
            for chain in community_chains:
                self.assertEqual(len(chain), 3)
    
    def test_chain_validity(self):
        """Test if the extracted chains are valid within their communities."""
        chains = optimized_extract_community_chains(
            graphml_path=self.graph_path,
            communities_parquet_path=self.communities_path,
            chain_length=3,
            n_communities=1,
            n_samples_per_community=5
        )
        
        # Get the first community's chains
        community_chains = next(iter(chains.values()))
        G = nx.read_graphml(self.graph_path)
        
        # Check if chains are valid
        for chain in community_chains:
            # Check if all nodes in the chain are connected
            for i in range(len(chain) - 1):
                self.assertTrue(
                    nx.has_path(G, chain[i], chain[i + 1])
                )
            
            # Check if first and last nodes are not directly connected
            self.assertFalse(
                G.has_edge(chain[0], chain[-1])
            )
    
    def test_chain_uniqueness(self):
        """Test if the extracted chains are unique."""
        chains = optimized_extract_community_chains(
            graphml_path=self.graph_path,
            communities_parquet_path=self.communities_path,
            chain_length=3,
            n_communities=1,
            n_samples_per_community=5
        )
        
        # Get the first community's chains
        community_chains = next(iter(chains.values()))
        
        # Convert chains to sets of frozensets to check uniqueness regardless of order
        chain_sets = {frozenset(chain) for chain in community_chains}
        self.assertEqual(len(chain_sets), len(community_chains))
    
    def tearDown(self):
        """Clean up test data after each test method."""
        # Remove test files
        if os.path.exists(self.graph_path):
            os.remove(self.graph_path)
        if os.path.exists(self.communities_path):
            os.remove(self.communities_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

if __name__ == '__main__':
    unittest.main() 