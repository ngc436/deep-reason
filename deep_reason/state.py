from typing import TypedDict
from deep_reason.schemes import Triplet

class KgConstructionStateInput(TypedDict):
    chunks: list[str] # List of chunks to process

class KgConstructionState(TypedDict):
    completed_triplets: list[Triplet] # Final key we duplicate in outer state for Send() API
    found_terms: list[str] # List of terms that were found in the text
    used_instruments: list[str] # List of used instruments

