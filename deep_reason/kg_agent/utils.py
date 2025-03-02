from contextlib import contextmanager
from datetime import datetime
import json
import logging
import pandas as pd
import os
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC
from pydantic import BaseModel
import tiktoken
from transformers import PreTrainedTokenizerBase

from deep_reason.schemes import Chunk
from deep_reason.kg_agent.schemes import Triplet, AggregationInput


logger = logging.getLogger(__name__)


@contextmanager
def measure_time(name: Optional[str] = None):
    start_time = datetime.now()
    yield
    end_time = datetime.now()

    suffix = f"for {name}" if name else ""
    logger.info(f"Time taken {suffix}: {(end_time - start_time).total_seconds()} seconds")


def load_obliqa_dataset(obliqa_dir: str) -> List[Chunk]:
    all_chunks = []
    for fname in os.listdir(obliqa_dir):
        df = pd.read_json(f"{obliqa_dir}/{fname}", orient="records")
        for ix, row in df.iterrows():
            all_chunks.append(Chunk(text=row["Passage"], 
                                    chapter_name=str(row["PassageID"]), 
                                    document_id=row["DocumentID"], 
                                    order_id=ix))
    return all_chunks


class KGConstructionAgentException(Exception):
    pass


class AggregationHelper(ABC):
    tokenizer: tiktoken.Encoding | PreTrainedTokenizerBase
    context_window_length: int

    def _estimate_size_in_tokens(self, state: Dict[str, Any] | Any | str) -> str:
        if isinstance(state, BaseModel):
            text = state.model_dump_json()
        elif isinstance(state, str):
            text = state
        else:
            text = f"{state}"
    
        return len(self.tokenizer.encode(text))

    def _create_batch(self, remaining_items: List[Triplet], current_state: Dict[str, Any] | Any | str) -> Tuple[AggregationInput, List[Triplet]]:
        """Create a batch of items that fits within the context window"""
        
        # Calculate current state size
        state_size = self._estimate_size_in_tokens(current_state)       
        # Dynamically form the next batch based on current state size
        current_batch = []
        current_batch_size = 0
        max_batch_size = self.context_window_length - state_size
        
        # Make a copy of remaining items to avoid modifying the original during batch creation
        updated_remaining = remaining_items.copy()
        
        # Add items to the batch until we reach the size limit
        while updated_remaining and current_batch_size < max_batch_size:
            item = updated_remaining[0]
            item_size = self._estimate_size_in_tokens(item)
            
            if current_batch_size + item_size <= max_batch_size:
                current_batch.append(item)
                current_batch_size += item_size
                updated_remaining.pop(0)
            else:
                # This item would exceed the batch size limit
                break
        
        # If we couldn't fit any items, take at least one (necessary for progress)
        if not current_batch and updated_remaining:
            raise KGConstructionAgentException(
                "Cannot pack a single item into the batch due to too long context (most probably due to too long result generated on the previous iteration)"
            )
            
        return AggregationInput(items=current_batch, input=current_state), updated_remaining

