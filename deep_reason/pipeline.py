from deep_reason.tools import WebSearchTool
from typing import List
import pandas as pd
from langgraph.graph import StateGraph, START, END
from deep_reason.state import KgConstructionState
web_tool = WebSearchTool(
        agent_id=web_tool_id,
        stream_events=tool_streaming,
        model="/model",
        model_kwargs={
            "temperature": 0,
            "top_p": 0.95,
            "max_tokens": 1024,
            "openai_api_key": "token-abc123",
            "openai_api_base": llm_props.llm_serving_url
        },
        retrievers="yandex",
        tokenizer=llm_props.llm_tokenizer_path,
        fast_search=True,
        model_type=llm_props.llm_type,
        embeddings=embeddings,
        langfuse_handler=langfuse_handler,
        callback=tool_callback,
    )

class KgConstructionPipeline:
    def __init__(self, tools: List[str]):
        self.tools = [self._get_tool(tool) for tool in tools]

    def _get_tool(self, tool_name: str):
        if "web" in tool_name:
            return web_tool
        else:
            raise ValueError(f"Tool {tool_name} not found")


    def get_knowledge_graph(self, chunk_df: pd.DataFrame):
        ''' Compile the knowledge graph from a set of chunks'''


        kg_extractor = StateGraph(KgConstructionState, input=)



        # ???
        ontology_builder = StateGraph()
        pass

