# from deep_reason.tools.tools import WebSearchTool
from typing import List, Dict, Any, Optional
# import pandas as pd
import os
from langgraph.graph import StateGraph, START, END
from deep_reason.state import KgConstructionState, KgConstructionStateInput
from deep_reason.prompts.kg_prompts import KG_PROMPT_VAR1
from deep_reason.schemes import Chunk, Triplet
# from langchain_core.rate_limiters import InMemoryRateLimiter
from deep_reason.utils import VLLMChatOpenAI
from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
# from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.output_parsers import RetryOutputParser, PydanticOutputParser
from deep_reason.schemes import ChunkTripletsResult
from langchain_core.runnables import Runnable, RunnableLambda
from openai import APIConnectionError
from langchain_core.exceptions import OutputParserException
import logging
import asyncio

logger = logging.getLogger(__name__)


# web_tool = WebSearchTool(
#         agent_id=web_tool_id,
#         stream_events=tool_streaming,
#         model="/model",
#         model_kwargs={
#             "temperature": 0,
#             "top_p": 0.95,
#             "max_tokens": 1024,
#             "openai_api_key": "token-abc123",
#             "openai_api_base": llm_props.llm_serving_url
#         },
#         retrievers="yandex",
#         tokenizer=llm_props.llm_tokenizer_path,
#         fast_search=True,
#         model_type=llm_props.llm_type,
#         embeddings=embeddings,
#         langfuse_handler=langfuse_handler,
#         callback=tool_callback,
#     )

class KgConstructionPipeline:
    def __init__(self, **model_kwargs):
        print(os.environ[OPENAI_API_BASE])
        self.llm = VLLMChatOpenAI(
            model="/model",
            base_url=os.environ[OPENAI_API_BASE],
            api_key=os.environ[OPENAI_API_KEY],
            temperature=model_kwargs.temperature if 'temperature' in model_kwargs else 0.3,
            max_tokens=model_kwargs.max_tokens if 'max_tokens' in model_kwargs else 2048,
            # rate_limiter=InMemoryRateLimiter(
            #     requests_per_second=mparams.request_rate_limit, 
            #     max_bucket_size=mparams.request_rate_limit
            # )
        )
        self.max_retry_attempts = 3

    def _get_tool(self, tool_name: str):
        raise NotImplementedError("Not implemented")
        # if "web" in tool_name:
        #     return web_tool
        # else:
        #     raise ValueError(f"Tool {tool_name} not found")


    async def _node_agent_triplets(self, state):
        '''
        Extract triplets from a set of chunks using LLM
        '''
        
        # Initialize lists for collecting results
        all_triplets = []
        terms = []
        instruments = []

        # Add the triplet extraction instrument to the used instruments
        if "triplet_extraction" not in instruments:
            instruments.append("triplet_extraction")
        
        parser = PydanticOutputParser(pydantic_object=ChunkTripletsResult)
        
        # Create the prompt template
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    KG_PROMPT_VAR1.format(response_format_description="{response_format_description}")
                ),
                HumanMessagePromptTemplate.from_template(
                    "{chunk}"
                )
            ]
        )
        
        retry_planner_parser = RetryOutputParser.from_llm(
            parser=parser,
            llm=self.llm,
            prompt=PromptTemplate.from_template("{prompt}"),
            max_retries=self.max_retry_attempts,
        )

        # Define the triplet extraction chain
        triplet_chain = (
            prompt | 
            self.llm | 
            RunnableLambda(lambda x: self._do_parsing_retrying(x, retry_planner_parser), name='retry_planner_lambda')
        )
        
        # Access chunks directly from state object if it's a Pydantic model
        # or access using attribute notation for regular objects
        chunks = state.chunks if hasattr(state, 'chunks') else state.get("chunks", [])
        print(chunks[0])
        for chunk in chunks:
            try:
                # Extract triplets using LLM
                result = await self._run_chain_with_retries(
                    chain=triplet_chain,
                    chain_kwargs={
                        "chunk": chunk.text if hasattr(chunk, 'text') else chunk,
                        "response_format_description": parser.get_format_instructions()
                    },
                    max_retry_attempts=3
                )
                            
            except Exception as e:
                # Log the error but continue processing other chunks
                logger.warning(f"Error processing chunk: {e}")
        
        # Update the state with the results
        return KgConstructionState(
            chunks=chunks,
            completed_triplets=all_triplets,
            found_terms=terms,
            used_instruments=instruments
        )
    
    async def _run_chain_with_retries(self, chain: Runnable, chain_kwargs: Dict[str, Any], max_retry_attempts: Optional[int] = None) -> Optional[Any]:
        retry_attempts = max_retry_attempts or self.max_retry_attempts
        retry_delay = 2
        attempt = 0
        
        while attempt < retry_attempts:
            try:
                result = await chain.ainvoke(chain_kwargs)
                return result  # Return immediately on success
            except OutputParserException:
                logger.warning("Parsing error occurred. Interrupting execution.")
                raise
            except APIConnectionError:
                logger.warning(f"APIConnectionError. Attempt: {attempt + 1}/{retry_attempts}. Retrying after {retry_delay} seconds")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            except Exception as e:
                logger.warning(f"Unexpected error during chain execution: {str(e)}")
            
            attempt += 1
        
        logger.error(f"All {retry_attempts} retry attempts failed.")
        return None  # Return None after all retries fail
    
    
    def _do_parsing_retrying(self, llm_output, retry_parser):
        """Parse the LLM output into a structured format with retry capability"""
        try:
            # First try to get the format directly
            response_format_description = retry_parser.parser.get_format_instructions()
            parsed_response = retry_parser.parse_with_prompt(
                llm_output, 
                prompt=f"Format the triplets according to this format: {response_format_description}\n\nText to format: {llm_output}"
            )
            return parsed_response
        except Exception as e:
            logger.warning(f"Error parsing LLM output: {e}")
            # Return empty result if parsing fails
            return ChunkTripletsResult(triplets=[])
        
    
    async def get_knowledge_graph(self, chunks: List[Chunk]):
        ''' Compile the knowledge graph from a set of chunks'''

        # Create initial state with chunks

        kg_extractor = StateGraph(KgConstructionState)
        # initial triplets extraction
        kg_extractor.add_node("get_triplets", self._node_agent_triplets)
        
        # Add edge from START to get_triplets
        kg_extractor.add_edge(START, "get_triplets")
        
        # Add edge from get_triplets to END
        kg_extractor.add_edge("get_triplets", END)
        
        # Compile the graph
        workflow = kg_extractor.compile()
        
        # Execute the workflow
        result = await workflow.ainvoke(KgConstructionState(chunks=chunks))
        
        # Return the completed triplets
        return result["completed_triplets"]

