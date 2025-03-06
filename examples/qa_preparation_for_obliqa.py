import os
import json
import asyncio
from deep_reason.envs import OPENAI_API_BASE, OPENAI_API_KEY
from deep_reason.utils import VLLMChatOpenAI
from deep_reason.chains import build_chain
from langchain_core.output_parsers import PydanticOutputParser
from deep_reason.schemes import QAnswer
import pandas as pd
import logging
from langchain_core.runnables import RunnableConfig
logger = logging.getLogger(__name__)


llm = VLLMChatOpenAI(
        model="/model",
        base_url=os.environ[OPENAI_API_BASE],
        api_key=os.environ[OPENAI_API_KEY],
        temperature=0.3,
        max_tokens=8096
    )

RAG_PROMPT_SYSTEM = """
You are a helpful assistant that can answer questions about the provided passages. 
Answer should be based only on the passages provided.
Provide answer in the following format:
{response_format_description}
"""

RAG_PROMPT_HUMAN = """
Question: {question}

Passages:
{passages}
"""

async def get_answers(data: pd.DataFrame):
    # Create a chain that uses data["Passages"] as context and data["Question"] as the question
    parser = PydanticOutputParser(pydantic_object=QAnswer)
    chain = build_chain(llm, RAG_PROMPT_SYSTEM, RAG_PROMPT_HUMAN, parser)

    # print(data["Passages"].tolist())
    inputs = []
    for ix, ct in data.iterrows():
        input_dict = {
            "question": ct['Question'],
            "passages": ct['Passages'],
        }
        inputs.append(input_dict)
    
    logger.info(f"Processing {len(inputs)} inputs")
    results = await chain.abatch(inputs, return_exceptions=True, 
                                 config=RunnableConfig(max_concurrency=250))
    data["answer"] = [result.answer for result in results]
    # answer = await chain.ainvoke({"question": data["Question"].tolist(), "passages": data["Passages"].tolist()})
    return data


def get_passages(row):
    # print(row)
    return '\n'.join([f'Passage {i}: {p["Passage"]}' for i, p in enumerate(row)])

async def main(): 
    # load the dataset
    questions_passages = pd.read_json("datasets/ObliQA/ObliQA_train.json")
    # print(questions_passages.head())
    questions_passages["Passages"] = questions_passages["Passages"].apply(get_passages)
    results = await get_answers(questions_passages)
    print(results.head())
    results.to_json("datasets/ObliQA/ObliQA_train_answers.json", orient="records", lines=True)
    # example of json entry:
    # {'QuestionID': 'a10724b5-ad0e-4b69-8b5e-792aef214f86', 'Question': 'Under Rules 7.3.2 and 7.3.3, what are the two specific conditions related to the maturity of a financial instrument that would trigger a disclosure requirement?', 'Passages': [{'DocumentID': 11, 'PassageID': '7.3.4', 'Passage': 'Events that trigger a disclosure. For the purposes of Rules 7.3.2 and 7.3.3, a Person is taken to hold Financial Instruments in or relating to a Reporting Entity, if the Person holds a Financial Instrument that on its maturity will confer on him:\n(1)\tan unconditional right to acquire the Financial Instrument; or\n(2)\tthe discretion as to his right to acquire the Financial Instrument.\n'}], 'Group': 1}
    # convert json to pandas df with 
    # json questionID, Question, Passages: [{}], Group, answer


if __name__ == "__main__":
    asyncio.run(main())