from typing import TypedDict
from deep_reason.schemes import Triplet, Chunk

class KgConstructionStateInput(TypedDict):
    chunks: list  # Can be either list[str] or list[Chunk]

class KgConstructionState(TypedDict):
    chunks: list[Chunk]
    completed_triplets: list[Triplet] # Final key we duplicate in outer state for Send() API
    found_terms: list[str] # List of terms that were found in the text
    used_instruments: list[str] # List of used instruments

