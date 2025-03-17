import os
from langchain_community.graphs import MemgraphGraph
# from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from baseline.utils.custom_memgraph_qa import CustomMemgraphQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from deep_reason.utils import VLLMChatOpenAI
from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.utils import load_obliqa_dataset, load_books_mx_dataset
from langchain_core.documents import Document
import pandas as pd
import logging
from typing import List
import asyncio
from langchain_core.runnables import RunnableConfig
from deep_reason.schemes import Chunk
import openai
from langchain_core.prompts.prompt import PromptTemplate
import click
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to query a graph database.
Schema of the graph: 
{schema}
Instructions:
- Use only the provided relationship types and properties in the schema. Pay attantion to the connection.
- Do not use any other relationship types or properties that are not provided.
- Find the most relevant nodes and relationship to answer the question.
- Try to make cypher statement as simple as possible to extract more data as graph can be large and sparse
Note: Do not include any explanations or apologies in your responses or additional tags.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# How many people played in Top Gun?
MATCH (m:Movie {{name:"Top Gun"}})<-[:ACTED_IN]-()
RETURN count(*) AS numberOfActors

The question is:
{question}
Generated Cypher statement:"""

# Neo4j connection details
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASSWORD = "password"  # Replace with your actual password

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")

def merge_chunks(chunks: List[Chunk], chunks_merge_window=100, char_len_limit=20000):
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

# New function to process documents in smaller batches to avoid context limit
async def process_documents_in_batches(llm, documents, batch_size=5):
    """
    Process documents in smaller batches to avoid context length errors.
    """
    graph_documents = []
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        try:
            batch_results = await llm_transformer.aconvert_to_graph_documents(
                batch, 
                config=RunnableConfig(max_concurrency=20)
            )
            graph_documents.extend(batch_results)
        except openai.BadRequestError as e:
            if "maximum context length" in str(e):
                print(f"Context length error in batch {i//batch_size + 1}. Reducing batch size.")
                # Try processing with an even smaller batch size
                for single_doc in batch:
                    try:
                        single_result = await llm_transformer.aconvert_to_graph_documents(
                            [single_doc], 
                            config=RunnableConfig(max_concurrency=1)
                        )
                        graph_documents.extend(single_result)
                    except Exception as inner_e:
                        print(f"Error processing document: {inner_e}")
                        continue
            else:
                print(f"Error processing batch: {e}")
                continue
    
    return graph_documents

# Function to process queries in batches
async def process_queries_in_batches(chain, inputs, batch_size=10):
    """
    Process queries in smaller batches to avoid context length errors.
    """
    all_responses = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        print(f"Processing query batch {i//batch_size + 1}/{(len(inputs) + batch_size - 1)//batch_size}")
        try:
            batch_responses = await chain.abatch(
                batch, config=RunnableConfig(max_concurrency=250)
            )
            all_responses.extend(batch_responses)
        except Exception as e:
            print(f"Error processing query batch: {e}")
            # Try processing one by one
            for query in batch:
                try:
                    single_response = await chain.ainvoke(query)
                    all_responses.append(single_response)
                except Exception as inner_e:
                    print(f"Error processing query: {inner_e}")
                    all_responses.append({"error": str(inner_e)})
    
    return all_responses

async def create_knowledge_graph(dataset_path="datasets/tatneft/книги_муслимов_хисамов/books_chunks.json", llm=None, batch_size=5):
    """
    Creates a knowledge graph from the provided dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        llm: The language model to use for creating the graph
        batch_size: Number of documents to process at once
        
    Returns:
        tuple: (graph, graph_documents) - The created graph and the processed graph documents
    """
    # Initialize graph
    graph = MemgraphGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
    
    # Load and process chunks
    if 'obliqa' in dataset_path.lower():
        chunks = load_obliqa_dataset(obliqa_dir=dataset_path, file_idx=[1])
    elif 'книги_муслимов_хисамов' in dataset_path.lower():
        chunks = load_books_mx_dataset(dataset_path)
    else:
        raise ValueError(f"Unknown dataset path: {dataset_path}")
    chunks = merge_chunks(chunks)
    print(f"Total chunks after merging: {len(chunks)}")
    chunks = chunks[:100] # TODO: remove
    
    # Convert chunks to documents
    documents = [Document(page_content=chunk.text) for chunk in chunks]
    
    # Process documents in smaller batches
    graph_documents = await process_documents_in_batches(llm, documents, batch_size=batch_size)
    print(f"Processed {len(graph_documents)} graph documents")
    
    # Reset graph and add documents
    graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    graph.query("DROP GRAPH")
    graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")
    
    graph.add_graph_documents(graph_documents, include_source=True)
    
    return graph, graph_documents

async def query_knowledge_graph(graph, llm, questions_path="datasets/ObliQA/ObliQA_train.json", 
                              output_path="datasets/gen_results/qwen_72b/ObliQA_train_graph_answers_langchain.json",
                              batch_size=50):
    """
    Queries the knowledge graph with questions from the provided file.
    
    Args:
        graph: The Memgraph graph to query
        llm: The language model to use for querying
        questions_path: Path to the questions file
        output_path: Path to save the results
        batch_size: Number of questions to process at once
        
    Returns:
        DataFrame: The questions with answers
    """
    if graph is None:
        graph = MemgraphGraph(
            url=url, username=username, password=password, refresh_schema=True
        )
    else:
        graph.refresh_schema()


    # Create query chain
    chain = CustomMemgraphQAChain.from_llm(
        llm,
        graph=graph,
        model_name="qwen2.5-72b-instruct",
        allow_dangerous_requests=True,
        cypher_prompt=PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE),
        # return_direct=True,
        return_intermediate_steps=True,
    )
    
    # Load questions
    questions_df = pd.read_json(questions_path)
    inputs = []
    questions_subset = questions_df.head(10) # TODO: remove
    
    for ct in questions_subset['Question'].tolist():
        input_dict = {
            "query": ct,
        }
        inputs.append(input_dict)
    
    # Process queries in batches
    responses = await process_queries_in_batches(chain, inputs, batch_size=batch_size)
    
    # Process and display results
    for i, response in enumerate(responses):
        print(f"Question {i+1}: {inputs[i]['query']}")
        if isinstance(response, dict) and "error" in response:
            print(f"Error: {response['error']}")
        else:
            print(f"Answer: {response}")
        print("-" * 50)
    
    # Save results to JSON
    try:
        # Extract results while handling potential errors
        results = []
        for response in responses:
            if isinstance(response, dict):
                if "error" in response:
                    results.append({"result": f"Error: {response['error']}"})
                else:
                    results.append({"result": response.get("result", "No result available")})
            else:
                results.append({"result": "Invalid response format"})
        
        questions_subset['answer'] = [r["result"] for r in results]
        questions_subset.to_json(output_path, orient="records", lines=True)
        print(f"Results saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        
    return questions_subset


# tat datasets/tatneft/книги_муслимов_хисамов/books_chunks.json
# obliqa datasets/ObliQA/StructuredRegulatoryDocuments
@click.command()
@click.option('--build-graph', is_flag=True, help='Whether to build the knowledge graph', default=False)
@click.option('--dataset-to-graph-path', default="datasets/ObliQA/StructuredRegulatoryDocuments", 
              help='Path to the dataset to create the graph')
@click.option('--answer-queries', is_flag=True, help='Whether to answer the questions', default=True)
@click.option('--query-path', default="datasets/ObliQA/ObliQA_train_filtered_doc1.json", 
              help='Path to JSON file with questions')
@click.option('--output-path', default="datasets/gen_results/qwen_72b/tat_testq_graph_answers_langchain.json", 
              help='Path to save the results')
@click.option('--model-params', default=None, 
              help='JSON string with parameters for the VLLMChatOpenAI model')
@click.option('--batch-size', default=50, type=int, 
              help='Batch size to process data with model')
def main(build_graph, dataset_to_graph_path, answer_queries, query_path, output_path, model_params, batch_size):
    """
    Main function to run the knowledge graph creation and querying.
    
    Parameters are provided through command-line arguments.
    """
    # Parse model parameters if provided
    model_config = {
        "model": "/model",
        "base_url": os.environ[OPENAI_API_BASE],
        "api_key": os.environ[OPENAI_API_KEY],
        "temperature": 0.3,
        "max_tokens": 2048
    }
    
    if model_params:
        try:
            import json
            custom_params = json.loads(model_params)
            model_config.update(custom_params)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for model parameters: {model_params}")
    
    # Initialize LLM
    llm = VLLMChatOpenAI(**model_config)
    
    # Initialize graph (will be used even if not building graph from scratch)
    graph = MemgraphGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
    
    # Run the async functions
    async def run():
        nonlocal graph  # Add this line to access the outer graph variable
        
        if build_graph:
            graph, _ = await create_knowledge_graph(
                dataset_path=dataset_to_graph_path,
                llm=llm,
                batch_size=batch_size
            )
        
        # Query knowledge graph
        if answer_queries:
            results = await query_knowledge_graph(
                graph, 
                llm,
                questions_path=query_path,
                output_path=output_path,
                batch_size=batch_size
            )
            return results
    
    # Run the async function
    return asyncio.run(run())

if __name__ == "__main__":
    main()
