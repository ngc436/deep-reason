#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List
import json
from datetime import datetime
from transformers import AutoTokenizer

from deep_reason.topic_modeling.bertopic.topic_modeling import (
    extract_topics_from_documents,
    create_topic_model,
    QwenEmbeddings
)

def chunk_text(text: str, max_tokens: int, tokenizer) -> List[str]:
    """Split text into chunks based on token limit using the specified tokenizer."""
    tokens = tokenizer.encode(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        if current_length + 1 <= max_tokens:
            current_chunk.append(token)
            current_length += 1
        else:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = [token]
            current_length = 1
    
    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))
    
    return chunks

def load_documents_from_folder(folder_path: str, max_tokens: int = 65536, tokenizer_name: str = "gpt2") -> List[str]:
    """Load all .txt documents from the specified folder and chunk them if needed."""
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder {folder_path} does not exist")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    for file_path in folder.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if max_tokens > 0:
                    chunks = chunk_text(content, max_tokens, tokenizer)
                    documents.extend(chunks)
                else:
                    documents.append(content)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    if not documents:
        raise ValueError(f"No .txt files found in {folder_path}")
    
    return documents

def main():
    parser = argparse.ArgumentParser(description="Topic Modeling CLI")
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing .txt documents"
    )
    parser.add_argument(
        "--api-base",
        type=str,
        required=True,
        help="API base URL for the embedding model"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key for the embedding model"
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=None,
        help="Number of topics to extract (default: automatic detection)"
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        help="Minimum size of topics (default: 10)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65536,
        help="Maximum number of tokens per document chunk (default: 65536, set to 0 for no chunking)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Name of the tokenizer model to use (default: gpt2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="topic_model_output",
        help="Directory to save output files (default: topic_model_output)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load documents
    print(f"Loading documents from {args.input_folder}...")
    documents = load_documents_from_folder(
        args.input_folder, 
        args.max_tokens,
        args.tokenizer
    )
    print(f"Loaded {len(documents)} document chunks")
    
    # Initialize embedding model with provided API parameters
    embedding_model = QwenEmbeddings(
        api_base=args.api_base,
        api_key=args.api_key
    )
    
    # Create topic model
    print("Creating topic model...")
    model = create_topic_model(
        embedding_model=embedding_model,
        n_topics=args.n_topics,
        min_topic_size=args.min_topic_size,
        documents=documents
    )
    
    # Extract topics
    print("Extracting topics...")
    result = extract_topics_from_documents(
        documents=documents,
        model=model,
        min_topic_size=args.min_topic_size,
        n_topics=args.n_topics
    )
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save topic information to JSON
    topic_output = {
        "topic_info": result["topic_info"].to_dict() if hasattr(result["topic_info"], 'to_dict') else result["topic_info"],
        "topic_representations": result["topic_representations"],
        "document_info": result["document_info"]
    }
    
    output_file = output_dir / f"topic_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(topic_output, f, indent=2)
    
    # Print summary
    print("\nTopic Modeling Results:")
    print(f"Number of document chunks processed: {len(documents)}")
    print(f"Number of topics found: {len(result['topic_representations'])}")
    print(f"\nTopics:")
    for topic_id, words in result["topic_representations"].items():
        print(f"Topic {topic_id}: {words}")
    
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    main() 