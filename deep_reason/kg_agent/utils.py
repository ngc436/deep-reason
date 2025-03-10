from contextlib import contextmanager
from datetime import datetime
import json
import logging
import pandas as pd
import os
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
from abc import ABC
from pydantic import BaseModel
import tiktoken
from transformers import PreTrainedTokenizerBase
from pathlib import Path
import hashlib

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


def load_obliqa_dataset(obliqa_dir: str, file_idx: None | List[int] = None) -> List[Chunk]:
    if file_idx is None:
        fnames = os.listdir(obliqa_dir)
    else:
        fnames = [f"{i}.json" for i in file_idx]
    all_chunks = []
    for fname in fnames:
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

T = TypeVar("T", bound=BaseModel)

class CacheManager:
    """
    A cache manager for storing and retrieving Pydantic models as JSON files on disk.
    Uses hashing to identify and retrieve cached results.
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _hash_input(self, input_data: Any) -> str:
        """Create a hash from input data."""
        if isinstance(input_data, BaseModel):
            input_str = input_data.model_dump_json(exclude_none=True)
        elif isinstance(input_data, list) and all(isinstance(item, BaseModel) for item in input_data):
            input_str = json.dumps([item.model_dump(exclude_none=True) for item in input_data])
        elif isinstance(input_data, dict):
            input_str = json.dumps(input_data, sort_keys=True)
        elif isinstance(input_data, (str, bytes)):
            input_str = str(input_data)
        else:
            input_str = str(input_data)
        
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, prefix: str = "") -> Path:
        """Get the file path for a cache key with optional prefix directory."""
        if prefix:
            prefix_dir = self.cache_dir / prefix
            os.makedirs(prefix_dir, exist_ok=True)
            return prefix_dir / f"{cache_key}.json"
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, input_data: Any, model_class: Type[T], prefix: str = "") -> Optional[T]:
        """
        Retrieve cached data if it exists.
        
        Args:
            input_data: The input data that was used to generate the cached result
            model_class: The Pydantic model class for deserializing the result
            prefix: Optional subdirectory to organize cache files
            
        Returns:
            The cached model instance or None if not found
        """
        cache_key = self._hash_input(input_data)
        cache_path = self._get_cache_path(cache_key, prefix)
        
        if cache_path.exists():
            try:
                logger.info(f"Cache hit for key {cache_key} in {prefix}")
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                return model_class.model_validate(cached_data)
            except Exception as e:
                logger.error(f"Error reading cache file {cache_path}: {e}")
                return None
        
        logger.info(f"Cache miss for key {cache_key} in {prefix}")
        return None
    
    def put(self, input_data: Any, result: T, prefix: str = "") -> None:
        """
        Store data in the cache.
        
        Args:
            input_data: The input data used to generate the result
            result: The Pydantic model instance to cache
            prefix: Optional subdirectory to organize cache files
        """
        if not isinstance(result, BaseModel):
            raise ValueError("Only Pydantic models can be cached")
            
        cache_key = self._hash_input(input_data)
        cache_path = self._get_cache_path(cache_key, prefix)
        
        try:
            with open(cache_path, 'w') as f:
                f.write(result.model_dump_json(exclude_none=True))
            logger.info(f"Cached result for key {cache_key} in {prefix}")
        except Exception as e:
            logger.error(f"Error writing to cache file {cache_path}: {e}")

