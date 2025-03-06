import os
# from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_community.graphs import MemgraphGraph
from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from deep_reason.utils import VLLMChatOpenAI
from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.kg_agent.utils import load_obliqa_dataset
from langchain_core.documents import Document
import pandas as pd
import logging
from typing import List
import asyncio
from langchain_core.runnables import RunnableConfig
from deep_reason.kg_agent.utils import Chunk
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Neo4j connection details
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASSWORD = "password"  # Replace with your actual password

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")

def merge_chunks(chunks: List[Chunk], chunks_merge_window=10, char_len_limit=20000):
    # Group chunks by document_id
    document_groups = {}
    for chunk in chunks:
        if chunk.document_id not in document_groups:
            document_groups[chunk.document_id] = []
        document_groups[chunk.document_id].append(chunk)
    
    # Sort each group by order_id
    for doc_id in document_groups:
        document_groups[doc_id].sort(key=lambda x: x.order_id)
    
    merged_chunks = []
    
    # Process each document group separately
    for doc_id, doc_chunks in document_groups.items():
        current_chunks = []
        current_text_length = 0
        current_batch = 0
        
        for i, chunk in enumerate(doc_chunks):
            # Check if adding this chunk would exceed char_len_limit
            # or if we've reached the chunks_merge_window limit
            chunk_text_length = len(chunk.text)
            
            if (current_text_length + chunk_text_length + (1 if current_chunks else 0) >= char_len_limit or  # +1 for space if not first chunk
                len(current_chunks) >= chunks_merge_window):
                
                if current_chunks:  # Only create a merged chunk if we have chunks to merge
                    # Combine text from all chunks in this group
                    combined_text = " ".join(chunk.text for chunk in current_chunks)
                    print(len(combined_text))
                    
                    # Create merged chunk
                    merged_chunk = Chunk(
                        document_id=doc_id,
                        order_id=current_batch,
                        text=combined_text,
                        chapter_name=None
                    )
                    merged_chunks.append(merged_chunk)
                    
                    # Reset for next batch
                    current_chunks = []
                    current_text_length = 0
                    current_batch += 1
            
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

    graph = MemgraphGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
        
    # text = """
    # Application\nThis Rulebook shall apply to Captive Insurers subject to alternative provision in these Rules or where the context otherwise requires.
    # """

 
    chunks = load_obliqa_dataset(obliqa_dir="datasets/ObliQA/StructuredRegulatoryDocuments")
    chunks = merge_chunks(chunks)
    print(len(chunks))

    documents = [Document(page_content=chunk.text) for chunk in chunks]
    # print(documents)

    llm = VLLMChatOpenAI(
            model="/model",
            base_url=os.environ[OPENAI_API_BASE],
            api_key=os.environ[OPENAI_API_KEY],
            temperature=0.3,
            max_tokens=6096
        )

    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = await llm_transformer.aconvert_to_graph_documents(documents, 
                                                                        config=RunnableConfig(max_concurrency=100))
    # print(f"Graph:{graph_documents}")

    graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    graph.query("DROP GRAPH")
    graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")


    graph.add_graph_documents(graph_documents, include_source=True)
    # graph.refresh_schema()
    # print(graph.get_schema)

    # query 
    chain = MemgraphQAChain.from_llm(
        llm,
        graph=graph,
        model_name="qwen2.5-72b-instruct",
        allow_dangerous_requests=True,
        # return_direct=True,
        return_intermediate_steps=True,
        # return_source_documents=True
    )

    # run all quesitons from the dataset

    questions_df = pd.read_json("datasets/ObliQA/ObliQA_train.json")
    inputs = []
    questions_subset = questions_df
    for ct in questions_subset['Question'].tolist():
        input_dict = {
            "query": ct,
        }
        inputs.append(input_dict)
    response = await chain.abatch(
        inputs, config=RunnableConfig(max_concurrency=250)
    )
    print(response)
    questions_subset['answer'] = [i['result'] for i in response]
    # questions_subset['intermediate_steps'] = [i['intermediate_steps'] for i in response]
    # questions_df['source_documents'] = [i['source_documents'] for i in response]
    questions_subset.to_json("datasets/gen_results/qwen_72b/ObliQA_train_graph_answers_langchain.json", 
                    orient="records", lines=True)

if __name__ == "__main__":
    asyncio.run(main())
