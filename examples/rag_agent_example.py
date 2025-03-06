import os
import logging
from typing import List

from deep_reason.utils import VLLMChatOpenAI
from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.kg_agent.utils import load_obliqa_dataset
from deep_reason.schemes import Chunk
from deep_reason.rag_agent.agent import RAGAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_chunks(chunks: List[Chunk], chunks_merge_window=10, char_len_limit=20000):
    """Merge chunks to optimize for RAG retrieval.
    
    This function is borrowed from examples/langchain_baseline.py.
    
    Args:
        chunks: List of chunks to merge
        chunks_merge_window: Number of chunks to merge together
        char_len_limit: Maximum character length for a merged chunk
        
    Returns:
        List of merged chunks
    """
    # Group chunks by document_id
    document_groups = {}
    for chunk in chunks:
        if chunk.document_id not in document_groups:
            document_groups[chunk.document_id] = []
        document_groups[chunk.document_id].append(chunk)
    
    # Sort chunks within each document by order_id
    for doc_id in document_groups:
        document_groups[doc_id].sort(key=lambda x: x.order_id)
    
    # Merge chunks within each document
    merged_chunks = []
    for doc_id, doc_chunks in document_groups.items():
        current_batch = 0
        current_chunks = []
        current_text_length = 0
        
        for i, chunk in enumerate(doc_chunks):
            chunk_text_length = len(chunk.text) if chunk.text else 0
            
            # If adding this chunk would exceed the character limit, create a merged chunk with current chunks
            if current_chunks and current_text_length + chunk_text_length > char_len_limit:
                combined_text = " ".join(chunk.text for chunk in current_chunks)
                merged_chunk = Chunk(
                    document_id=doc_id,
                    order_id=current_batch,
                    text=combined_text,
                    chapter_name=None
                )
                merged_chunks.append(merged_chunk)
                current_batch += 1
                current_chunks = []
                current_text_length = 0
            
            # Handle the case where a single chunk exceeds the char limit
            if chunk_text_length >= char_len_limit:
                # Create a separate chunk for this large text
                merged_chunk = Chunk(
                    document_id=doc_id,
                    order_id=current_batch,
                    text=chunk.text[:char_len_limit-1],  # Truncate to fit limit
                    chapter_name=None
                )
                merged_chunks.append(merged_chunk)
                current_batch += 1
                continue
            
            # Add current chunk to the batch
            current_chunks.append(chunk)
            current_text_length += chunk_text_length + (1 if current_chunks else 0)  # +1 for space if not first chunk
            
            # If this is the last chunk, create a merged chunk with remaining chunks
            if i == len(doc_chunks) - 1 and current_chunks:
                combined_text = " ".join(chunk.text for chunk in current_chunks)
                merged_chunk = Chunk(
                    document_id=doc_id,
                    order_id=current_batch,
                    text=combined_text,
                    chapter_name=None
                )
                merged_chunks.append(merged_chunk)
    
    return merged_chunks


async def main():
    """Main function to demonstrate the RAG agent."""
    # Initialize the language model
    llm = VLLMChatOpenAI(
        model="/model",
        base_url=os.environ[OPENAI_API_BASE],
        api_key=os.environ[OPENAI_API_KEY],
        temperature=0.3,
        max_tokens=2048
    )
    
    # Create the RAG agent
    rag_agent = RAGAgent(
        llm=llm,
        elasticsearch_index="obliqa_documents",
        elasticsearch_url="http://localhost:9200"  # Change as needed
    )
    
    # Load and preprocess documents
    logger.info("Loading documents...")
    chunks = load_obliqa_dataset(obliqa_dir="datasets/ObliQA/StructuredRegulatoryDocuments")
    chunks = merge_chunks(chunks)
    logger.info(f"Loaded and merged {len(chunks)} document chunks")
    
    # Index the documents
    logger.info("Indexing documents...")
    result = await rag_agent.index_documents(chunks)
    logger.info(f"Indexing result: {result}")
    
    # Example questions
    questions = [
        "What are the requirements for Captive Insurers?",
        "What is the procedure for applying for a license?",
        "What are the capital requirements for insurance companies?"
    ]
    
    # Answer questions
    for question in questions:
        logger.info(f"Question: {question}")
        answer = await rag_agent.query(question)
        logger.info(f"Answer: {answer['answer']}")
        logger.info(f"Reasoning: {answer['reasoning']}")
        logger.info("---")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 