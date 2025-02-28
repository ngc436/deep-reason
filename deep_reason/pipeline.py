from deep_reason.tools.tools import WebSearchTool
from typing import List
import pandas as pd
from langgraph.graph import StateGraph, START, END
from deep_reason.state import KgConstructionState, KgConstructionStateInput
from deep_reason.prompts.kg_prompts import KG_PROMPT_VAR1
from deep_reason.schemes import Chunk, Triplet

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
    def __init__(self):
        pass
        # self.tools = [self._get_tool(tool) for tool in tools]

    def _get_tool(self, tool_name: str):
        raise NotImplementedError("Not implemented")
        # if "web" in tool_name:
        #     return web_tool
        # else:
        #     raise ValueError(f"Tool {tool_name} not found")

    async def _node_aget_triplets(self, 
                                  state: KgConstructionState):
        ''' Get triplets from a set of chunks'''
        pass

    async def _node_agent_triplets(self, state: KgConstructionState):
        '''
        Extract triplets from a set of chunks using LLM
        Uses KG_PROMPT_VAR1 to prompt the LLM to extract triplets
        '''
        from langchain_openai import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.output_parsers import StrOutputParser
        
        # Initialize lists for collecting results
        all_triplets = state.get("completed_triplets", [])
        terms = state.get("found_terms", [])
        instruments = state.get("used_instruments", [])
        
        # Add the triplet extraction instrument to the used instruments
        if "triplet_extraction" not in instruments:
            instruments.append("triplet_extraction")
        
        # Initialize LLM from VLLM
        llm = ChatOpenAI(
            model="gpt-4",  # You should replace with your preferred model
            temperature=0,
            max_tokens=1024
        )
        
        # Create the prompt template
        prompt = PromptTemplate(
            template=KG_PROMPT_VAR1,
            input_variables=["example", "observation"]
        )
        
        # Example triplets to provide as an example
        example_triplets = (
            "apple, is a, fruit; apple, grows on, tree; "
            "tree, has, leaves; tree, requires, water"
        )
        
        # Define the triplet extraction chain
        triplet_chain = (
            prompt | 
            llm | 
            StrOutputParser()
        )
        
        # Process each chunk
        chunks = state.get("chunks", [])
        for chunk in chunks:
            try:
                # Extract triplets using LLM
                triplet_text = await triplet_chain.ainvoke({
                    "example": example_triplets,
                    "observation": chunk.text if isinstance(chunk, Chunk) else chunk
                })
                
                # Parse the triplets from the text response
                # Format should be "subject, relation, object; subject, relation, object; ..."
                triplet_strings = triplet_text.split(";")
                
                for triplet_str in triplet_strings:
                    triplet_str = triplet_str.strip()
                    if not triplet_str:
                        continue
                        
                    parts = [part.strip() for part in triplet_str.split(",", 2)]
                    if len(parts) == 3:
                        subject, relation, obj = parts
                        all_triplets.append(Triplet(
                            subject=subject,
                            relation=relation,
                            object=obj
                        ))
                        
                        # Add subject and object to found terms
                        if subject not in terms:
                            terms.append(subject)
                        if obj not in terms:
                            terms.append(obj)
                            
            except Exception as e:
                # Log the error but continue processing other chunks
                print(f"Error processing chunk: {e}")
        
        # Update the state with the results
        return {
            "completed_triplets": all_triplets,
            "found_terms": terms,
            "used_instruments": instruments
        }

    def get_knowledge_graph(self, chunks: List[Chunk]):
        ''' Compile the knowledge graph from a set of chunks'''

        # transform input to the correct List of Chunk format

        kg_extractor = StateGraph(KgConstructionState, 
                                  input=KgConstructionStateInput)
        # initial triplets extraction
        kg_extractor.add_node("get_triplets", self._node_agent_triplets)
        
        # Add edge from START to get_triplets
        kg_extractor.add_edge(START, "get_triplets")
        
        # Add edge from get_triplets to END
        kg_extractor.add_edge("get_triplets", END)
        
        # Compile the graph
        workflow = kg_extractor.compile()
        
        # Execute the workflow
        result = workflow.invoke({"chunks": chunks})
        
        # Return the completed triplets
        return result["completed_triplets"]

