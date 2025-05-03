import numpy as np
from typing import List, Dict, Optional
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, BaseRepresentation
import openai
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

class QwenRepresentation(BaseRepresentation):
    def __init__(self, 
                 api_base: str = "http://10.32.2.11:8164/v1", 
                 api_key: str = "token-abc123",
                 model: str = "/model",
                 top_n_words: int = 10):
        super().__init__()
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model = model
        self.top_n_words = top_n_words

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        """Extract topics using Qwen3 model"""
        topic_representations = {}
        
        for topic_id in set(topics):
            if topic_id != -1:  # Skip outlier topic
                # Get documents for this topic
                topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]
                
                # Create prompt for topic representation
                prompt = f"""Given the following documents, extract the {self.top_n_words} most important and representative words that describe their common topic:
                
                Documents:
                {' '.join(topic_docs[:3])}  # Use first 3 documents for context
                
                Please list the words in order of importance, separated by commas."""
                
                # Get response from Qwen3
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                
                # Extract words from response
                words = response.choices[0].message.content.strip().split(',')
                words = [word.strip() for word in words]
                
                # Format as BERTopic expects: list of (word, score) tuples
                topic_representations[topic_id] = [(word, 1.0 - (i/len(words))) for i, word in enumerate(words)]
        
        return topic_representations

# Configure the embedding model - qwen embeddings
class QwenEmbeddings:
    def __init__(self, 
                 api_base="http://d.dgx:8029/v1", 
                 api_key="token-abc123"):
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
            
        embeddings = []
        # Process in batches to avoid potential API limitations
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model="embedding-model" # Assuming default model, modify if needed
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)

def create_topic_model(
    embedding_model: Optional[object] = None,
    n_topics: Optional[int] = None,
    min_topic_size: int = 10,
    verbose: bool = True,
    documents: Optional[List[str]] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None
) -> BERTopic:
    """
    Create a BERTopic model configured with the specified embedding and language models.
    
    Args:
        embedding_model: Custom embedding model (if None, will use Qwen embeddings)
        n_topics: Number of topics to extract (None for automatic detection)
        min_topic_size: Minimum size of topics
        verbose: Whether to print progress information
        documents: List of documents to be processed (used to set appropriate parameters)
        api_base: API base URL for the language model
        api_key: API key for the language model
        
    Returns:
        Configured BERTopic model
    """
    # Use qwen embeddings if no custom model is provided
    if embedding_model is None:
        embedding_model = QwenEmbeddings()
    
    # Configure UMAP for better dimensionality reduction
    umap_model = UMAP(
        n_neighbors=5,  # Reduced further for more localized clusters
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        low_memory=True,
        densmap=True  # Enable density-aware UMAP
    )
    
    # Configure HDBSCAN for better topic control
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,  # Reduced from 10 to allow more topics
        metric='euclidean',
        cluster_selection_method='eom',  # Changed back to 'eom' for better cluster selection
        prediction_data=True,
        min_samples=5,  # Increased for better stability
        cluster_selection_epsilon=0.1,  # Reduced for more clusters
        allow_single_cluster=False,  # Changed to prevent single large cluster
        alpha=1.0,
        core_dist_n_jobs=1
    )
    
    # Configure vectorizer for Russian text
    vectorizer_model = CountVectorizer(
        stop_words=None,  # Don't use English stopwords
        min_df=2,  # Lower min_df to allow more terms
        max_df=0.8,  # Lower max_df to filter out very common terms
        max_features=10000,  # Maximum number of features
        ngram_range=(1, 2),  # Include both single words and bigrams
        token_pattern=r'(?u)\b\w+\b'  # Include all word characters
    )
    
    # Configure representation model
    if api_base and api_key:
        representation_model = QwenRepresentation(
            api_base=api_base,
            api_key=api_key,
            model="/model"
        )
    else:
        # Use KeyBERT and MMR for representation with adjusted parameters
        representation_model = {
            "KeyBERT": KeyBERTInspired(
                top_n_words=10,
                nr_repr_docs=5
            ),
            "MMR": MaximalMarginalRelevance(
                diversity=0.3  # Lower diversity to keep more relevant words
            )
        }
    
    # Create and return BERTopic model with adjusted parameters
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=n_topics,
        verbose=verbose,
        calculate_probabilities=True,
        min_topic_size=min_topic_size
    )
    
    return topic_model

def extract_topics_from_documents(
    documents: List[str],
    model: Optional[BERTopic] = None,
    min_topic_size: int = 10,
    n_topics: Optional[int] = None
) -> Dict:
    """
    Extract topics from a list of documents
    
    Args:
        documents: List of documents to analyze
        model: BERTopic model (if None, will create a new one)
        min_topic_size: Minimum size of topics
        n_topics: Number of topics to extract (None for automatic)
        
    Returns:
        Dictionary containing topics, topic model, and other metadata
    """
    if model is None:
        model = create_topic_model(min_topic_size=min_topic_size, n_topics=n_topics, documents=documents)
    
    # Fit the model and transform documents
    topics, probs = model.fit_transform(documents)
    
    # Get the topic info
    topic_info = model.get_topic_info()
    
    # Get topic representations
    topic_representations = {}
    for topic_id in set(topics):
        if topic_id != -1:  # Skip outlier topic
            topic_representations[topic_id] = model.get_topic(topic_id)
    
    # Get document info
    document_info = []
    for i, (doc, topic, prob) in enumerate(zip(documents, topics, probs)):
        document_info.append({
            "document_id": i,
            "document": doc[:100] + "..." if len(doc) > 100 else doc,  # Truncate for readability
            "topic": topic,
            "probability": prob
        })
    
    return {
        "model": model,
        "topics": topics,
        "probabilities": probs,
        "topic_info": topic_info,
        "topic_representations": topic_representations,
        "document_info": document_info
    }

def load_dataset_obliqa(file_path: str) -> List[str]:
    """
    Load documents from the obliqa dataset format.
    
    Args:
        file_path: Path to the obliqa dataset file
        
    Returns:
        List of documents extracted from the dataset
    """
    # Placeholder implementation - adjust based on actual file format
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Assume each line is a JSON object with a 'text' field
                # Modify this according to the actual format of obliqa dataset
                import json
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        documents.append(data['text'])
                except json.JSONDecodeError:
                    # If not JSON, treat the line as a document
                    documents.append(line.strip())
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    return documents

def infer_topics_for_new_documents(
    model: BERTopic,
    documents: List[str]
) -> Dict:
    """
    Infer topics for new documents using a pre-trained model
    
    Args:
        model: Pre-trained BERTopic model
        documents: List of new documents to analyze
        
    Returns:
        Dictionary containing inferred topics and probabilities
    """
    topics, probs = model.transform(documents)
    
    # Get document info
    document_info = []
    for i, (doc, topic, prob) in enumerate(zip(documents, topics, probs)):
        document_info.append({
            "document_id": i,
            "document": doc[:100] + "..." if len(doc) > 100 else doc,  # Truncate for readability
            "topic": topic,
            "probability": prob
        })
    
    return {
        "topics": topics,
        "probabilities": probs,
        "document_info": document_info
    }

def visualize_topics(model: BERTopic, output_file: Optional[str] = None):
    """
    Generate and optionally save a visualization of topics
    
    Args:
        model: BERTopic model
        output_file: Path to save the visualization (None to display only)
    """
    fig = model.visualize_topics()
    if output_file:
        fig.write_html(output_file)
    return fig

# Example usage
if __name__ == "__main__":
    # Example documents
    documents = [
        "This is a document about artificial intelligence and machine learning.",
        "Neural networks have revolutionized computer vision tasks.",
        "Natural language processing enables computers to understand human language.",
        "Climate change is affecting global weather patterns.",
        "Renewable energy sources are becoming more affordable."
    ]
    
    # Extract topics
    result = extract_topics_from_documents(documents)
    
    # Print topics
    print("Topics extracted:")
    for topic_id, words in result["topic_representations"].items():
        print(f"Topic {topic_id}: {words}")
    
    # Example of inferring topics for new documents
    new_documents = [
        "Deep learning models require large amounts of training data.",
        "Solar and wind power are examples of renewable energy."
    ]
    
    infer_result = infer_topics_for_new_documents(result["model"], new_documents)
    print("\nInferred topics for new documents:")
    for doc_info in infer_result["document_info"]:
        print(f"Document: {doc_info['document']}")
        print(f"Topic: {doc_info['topic']}")
