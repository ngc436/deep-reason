import os
# from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_community.graphs import MemgraphGraph
from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from deep_reason.utils import VLLMChatOpenAI
from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.kg_agent.utils import load_obliqa_dataset
from langchain_core.documents import Document
import logging
import asyncio
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

async def main():
    # First check if APOC is available
    url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    username = os.environ.get("MEMGRAPH_USERNAME", "")
    password = os.environ.get("MEMGRAPH_PASSWORD", "")

    graph = MemgraphGraph(
        url=url, username=username, password=password, refresh_schema=False
    )
        
    # text = """
    # Application\nThis Rulebook shall apply to Captive Insurers subject to alternative provision in these Rules or where the context otherwise requires.
    # """

    chunks = load_obliqa_dataset(obliqa_dir="datasets/ObliQA/StructuredRegulatoryDocuments")

    documents = [Document(page_content=chunk.text) for chunk in chunks[:100]]
    # print(documents)

    llm = VLLMChatOpenAI(
            model="/model",
            base_url=os.environ[OPENAI_API_BASE],
            api_key=os.environ[OPENAI_API_KEY],
            temperature=0.3,
            max_tokens=8096
        )

    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
    print(f"Graph:{graph_documents}")

    graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
    graph.query("DROP GRAPH")
    graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")


    graph.add_graph_documents(graph_documents, include_source=True)
    # graph.refresh_schema()
    print(graph.get_schema)

    # query 
    chain = MemgraphQAChain.from_llm(
        llm,
        graph=graph,
        model_name="qwen2.5-72b-instruct",
        allow_dangerous_requests=True,
        # return_direct=True,
        return_intermediate_steps=True
    )

    response = chain.invoke(
        "common law is based on what?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
