from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from typing import Any, Dict, List, Optional
import logging
import asyncio
from pydantic import BaseModel, Field

from deep_reason.schemes import Chunk
from deep_reason.utils import build_chain

logger = logging.getLogger(__name__)

# TODO: Make implementation of RAG agent the same way as for kg_agent
# rag agent should have a separate chain for encoding loading data to elasticsearch
# retrieval chain should find the most relevant chunks for the query 
# and then pass them to the LLM for answering the query

# RAG agent implementation based on kg_agent pattern
# Implementation includes encoding/loading data to elasticsearch and retrieval chain

class RAGDocumentResponse(BaseModel):
    """Response format for document search and answer generation"""
    answer: str = Field(description="The answer to the query based on the documents")
    context_chunks: List[str] = Field(description="The list of chunks that were used to answer the query")
    reasoning: str = Field(description="The reasoning behind the answer")


class ElasticsearchDocumentEncoder(Runnable):
    """Encodes and loads documents into Elasticsearch for retrieval."""
    
    def __init__(self, elasticsearch_index: str, elasticsearch_url: Optional[str] = None):
        """Initialize the encoder with Elasticsearch connection details.
        
        Args:
            elasticsearch_index: The name of the Elasticsearch index to use
            elasticsearch_url: The URL of the Elasticsearch instance
        """
        self.elasticsearch_index = elasticsearch_index
        self.elasticsearch_url = elasticsearch_url or "http://localhost:9200"
        # Import here to make elasticsearch an optional dependency
        from elasticsearch import Elasticsearch
        self.es_client = Elasticsearch(self.elasticsearch_url)
        
        # Create index if it doesn't exist
        if not self.es_client.indices.exists(index=self.elasticsearch_index):
            self.es_client.indices.create(
                index=self.elasticsearch_index,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "chapter_name": {"type": "keyword"},
                            "document_id": {"type": "integer"},
                            "order_id": {"type": "integer"},
                            "embedding": {"type": "dense_vector", "dims": 1536}  # Adjust dims based on your embedding model
                        }
                    }
                }
            )
    
    async def ainvoke(self, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Encode and load chunks into Elasticsearch asynchronously.
        
        Args:
            chunks: List of chunks to encode and load
            config: Optional configuration for the encoding process
            
        Returns:
            Dictionary with the result of the operation
        """
        logger.info(f"Encoding and loading {len(chunks)} chunks into Elasticsearch")
        
        # Import transformers here to make it an optional dependency
        from sentence_transformers import SentenceTransformer
        
        # Load the model for encoding
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Can be changed to any sentence-transformer model
        
        # Process chunks in batches
        batch_size = 32  # Adjust based on your hardware
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Generate embeddings for the batch
            texts = [chunk.text for chunk in batch]
            embeddings = model.encode(texts)
            
            # Prepare bulk operation
            operations = []
            for chunk, embedding in zip(batch, embeddings):
                operations.extend([
                    {"index": {"_index": self.elasticsearch_index}},
                    {
                        "text": chunk.text,
                        "chapter_name": chunk.chapter_name,
                        "document_id": chunk.document_id,
                        "order_id": chunk.order_id,
                        "embedding": embedding.tolist()
                    }
                ])
            
            # Execute bulk operation
            if operations:
                self.es_client.bulk(operations)
        
        logger.info(f"Successfully loaded {len(chunks)} chunks into Elasticsearch")
        return {"status": "success", "chunks_loaded": len(chunks)}
    
    def invoke(self, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version that runs the async method in an event loop."""
        return asyncio.run(self.ainvoke(chunks, config))


def build_retrieval_chain(llm: BaseChatModel, elasticsearch_index: str, elasticsearch_url: Optional[str] = None) -> Runnable:
    """Build a chain for retrieving relevant chunks from Elasticsearch and answering queries.
    
    Args:
        llm: The language model to use for answering
        elasticsearch_index: The Elasticsearch index to search
        elasticsearch_url: The URL of the Elasticsearch instance
        
    Returns:
        A runnable chain that retrieves and answers queries
    """
    from elasticsearch import Elasticsearch
    from sentence_transformers import SentenceTransformer
    
    es_client = Elasticsearch(elasticsearch_url or "http://localhost:9200")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Can be changed to any sentence-transformer model
    
    system_template = """You are a helpful assistant that accurately answers questions based on the provided context. 
    If the answer cannot be found in the context, say "I don't know" rather than making up an answer.
    Your goal is to provide accurate, concise, and helpful answers based solely on the information provided in the context.
    {response_format_description}"""

    human_template = """Answer the following question using only the context below:
    
Question: {query}

Context:
{context}

Answer the question based only on the provided context, citing relevant parts when appropriate. If the question cannot be
 answered from the context, just say "I don't know based on the provided information."
"""
    
    from langchain_core.output_parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=RAGDocumentResponse)
    
    qa_chain = build_chain(llm, system_template, human_template, parser)
    
    def _retrieve_context(query: str) -> Dict[str, Any]:
        """Retrieve the most relevant chunks from Elasticsearch for the query.
        
        Args:
            query: The query to search for
            
        Returns:
            Dictionary with the query and retrieved context
        """
        # Encode the query
        query_embedding = embedding_model.encode(query)
        
        # Search Elasticsearch for similar documents
        results = es_client.search(
            index=elasticsearch_index,
            body={
                "size": 5,  # Get top 5 most relevant chunks
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding.tolist()}
                        }
                    }
                }
            }
        )
        
        # Extract the chunks from the results
        chunks = []
        for hit in results["hits"]["hits"]:
            chunks.append(hit["_source"]["text"])
        
        context = "\n\n".join(chunks)
        
        return {
            "query": query,
            "context": context,
            "retrieved_chunks": chunks
        }
    
    def _process_result(result: RAGDocumentResponse, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the result of the QA chain.
        
        Args:
            result: The result of the QA chain
            inputs: The inputs to the QA chain
            
        Returns:
            Dictionary with the answer and retrieved chunks
        """
        return {
            "answer": result.answer,
            "context_chunks": inputs["retrieved_chunks"],
            "reasoning": result.reasoning
        }
    
    # Complete chain: retrieve context -> pass to QA chain -> process result
    retrieval_chain = (
        RunnableLambda(_retrieve_context)
        | RunnableParallel(
            result=qa_chain,
            inputs=RunnablePassthrough()
        )
        | RunnableLambda(lambda x: _process_result(x["result"], x["inputs"]))
    )
    
    return retrieval_chain