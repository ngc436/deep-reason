import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel


from deep_reason.schemes import Chunk
from deep_reason.rag_agent.chains import ElasticsearchDocumentEncoder, build_retrieval_chain

logger = logging.getLogger(__name__)


class RAGAgent:
    """Agent for Retrieval-Augmented Generation using Elasticsearch.
    
    This agent handles document encoding, indexing, and retrieval for question answering.
    It follows a similar pattern to the KGConstructionAgent.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        elasticsearch_index: str = "document_index",
        elasticsearch_url: Optional[str] = None,
    ):
        """Initialize the RAG agent.
        
        Args:
            llm: The language model to use for question answering
            elasticsearch_index: The name of the Elasticsearch index to use
            elasticsearch_url: The URL of the Elasticsearch instance
        """
        self.llm = llm
        self.elasticsearch_index = elasticsearch_index
        self.elasticsearch_url = elasticsearch_url or "http://localhost:9200"
        
        # Initialize the encoder and retrieval chains
        self.encoder = ElasticsearchDocumentEncoder(
            elasticsearch_index=elasticsearch_index,
            elasticsearch_url=elasticsearch_url
        )
        self.retrieval_chain = build_retrieval_chain(
            llm=llm,
            elasticsearch_index=elasticsearch_index,
            elasticsearch_url=elasticsearch_url
        )
    
    async def index_documents(self, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Index documents in Elasticsearch.
        
        Args:
            chunks: List of document chunks to index
            config: Optional configuration for the indexing process
            
        Returns:
            Status of the indexing operation
        """
        logger.info(f"Indexing {len(chunks)} document chunks")
        
        # Use the encoder to index the documents
        result = await self.encoder.ainvoke(chunks, config=config)
        
        logger.info(f"Indexing completed with status: {result['status']}")
        return result
    
    async def query(self, query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the indexed documents and retrieve an answer.
        
        Args:
            query: The query to search for
            config: Optional configuration for the retrieval process
            
        Returns:
            Dictionary with the answer, context chunks, and reasoning
        """
        logger.info(f"Processing query: {query}")
        
        # Use the retrieval chain to answer the query
        result = await self.retrieval_chain.ainvoke(query, config=config)
        
        logger.info(f"Query processed successfully")
        return result
    
    def index_documents_sync(self, chunks: List[Chunk], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version of index_documents."""
        import asyncio
        return asyncio.run(self.index_documents(chunks, config))
    
    def query_sync(self, query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous version of query."""
        import asyncio
        return asyncio.run(self.query(query, config)) 