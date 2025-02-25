import os
import logging
from typing import Annotated

import chromadb
from langchain_openai import ChatOpenAI

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

class KgAgent:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection("kg_collection")

    def add_entity(self, entity: dict):
        self.collection.add(entity)