from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from typing import Any, Dict, List, Optional
import logging

from deep_reason.schemes import Chunk
from deep_reason.utils import build_chain

logger = logging.getLogger(__name__)


class RAGAgent(Runnable):
    """Process chunks to extract knowledge triplets using sliding window context approach."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    async def ainvoke(self, questions: List[str], passages: List[str], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Takes questions and answer on them using passages as context.
        
        Args:
            questions: List of questions to answer
            passages: List of passages to use as context
            config: Optional configuration for the chain
            
        Returns:
            List of dictionaries containing question and answer information
        """
        logger.info(f"Sorting chunks by document_id and order_id for {len(chunks)} chunks")
        # 1. Order chunks by document_id and order_id
        chunks = sorted(chunks, key=lambda x: (x.document_id, x.order_id))
        
        # 2. Create chunk tuples with sliding window approach
        chunk_tuples = []
        logger.info(f"Creating chunks tuples by document_id and order_id for {len(chunks)} chunks")
        # Group chunks by document_id
        for _, doc_chunks in groupby(chunks, key=lambda x: x.document_id):
            doc_chunks_list = list(doc_chunks)
            
            for i, chunk in enumerate(doc_chunks_list):
                left_context = None if i == 0 else doc_chunks_list[i-1]
                right_context = None if i == len(doc_chunks_list) - 1 else doc_chunks_list[i+1]
                
                chunk_tuples.append(ChunkTuple(
                    current_chunk=chunk,
                    left_context=left_context,
                    right_context=right_context
                ))
        
        # 3. Build the triplet mining chain
        chain = build_triplets_mining_chain(self.llm)
        
        # 4. Process chunk tuples using batch method
        logger.info(f"Forming batches for {len(chunk_tuples)} chunk tuples")
        inputs = []
        for ct in chunk_tuples:
            input_dict = {
                "current_chunk": ct.current_chunk.text,
                "left_context_prefix": "Left context: " if ct.left_context else "No left context available.",
                "left_context": ct.left_context.text if ct.left_context else "",
                "right_context_prefix": "Right context: " if ct.right_context else "No right context available.",
                "right_context": ct.right_context.text if ct.right_context else ""
            }
            inputs.append(input_dict)
        
        logger.info(f"Processing {len(inputs)} inputs")
        results = await chain.abatch(inputs, return_exceptions=True, config=config)
        
        # 5. Handle exceptions
        valid_results = []
        for chunk_tuple, result in zip(chunk_tuples, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing chunk {chunk_tuple.current_chunk.chapter_name}: {result}. Skipping chunk.")
                continue
            # Attach the chunk information to each result
            for triplet in result.triplets:
                valid_results.append(
                    (chunk_tuple.current_chunk.document_id, chunk_tuple.current_chunk.order_id, triplet)
                )
        
        # 6. Sort triplets by document and then chunk order
        valid_results = [triplet for _, _, triplet in sorted(valid_results, key=lambda x: (x[0], x[1]))]
        
        return valid_results