from dataclasses import dataclass
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

keywords_extraction_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are given a question or statement. Your task is to form a list of search terms for searching in an ElasticSearch database.
Extract all important keywords, terms, and abbreviations from the following question and write them as a comma-separated list of words.
USE ONLY WORDS and WORD FORMS FROM THE GIVEN QUESTION.
DO NOT add words, terms, and abbreviations that are not in the original question and DO NOT answer the question itself!

Here are several examples of questions and generated lists:
Question: What is KPI?
Answer: KPI

Question: How is a high-pressure pump connected to an oil derrick?
Answer: connect, pump, high, pressure, derrick

Question: Give me a list of TTN rules
Answer: list, rules, TTN

Format your response in the following JSON format:
{response_format_description}

Respond only in English and use JSON format.
"""),
        ("user", "{question}")
    ]
)


planner_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a system that plans queries for retrieving user-requested information from a vector database with documents divided into small segments.
For each incoming user query, your task is to divide it, if necessary, into several queries to the vector database to obtain the information needed for the answer.
You should write the final queries under "Queries:", and then write an explanation under "Explanation:". Both queries and explanation should be in English.
For example, the question "what is the capital of Russia" doesn't need division, the necessary information about Russia's capital will be extracted.
On the other hand, the query "Compare Audi Q5 and Audi Q7" should be divided into two: "What are the advantages and disadvantages of Audi Q5" and "What are the advantages and disadvantages of Audi Q7",
since we need information about both objects for comparison.
Use the following set of rules:
1. Each subsequent question should not rely on the result of the previous answer (avoid: "Which of these technologies..."). Information aggregation will happen later.
2. Write generated questions in the same language as the user query.
3. Consider that the user query has context that cannot be lost.
4. If the user query can be divided into several, don't forget to add the necessary context for independent database search.

Format your response in the following JSON format:
{response_format_description}
"""),
        ("user", "{question}")
    ]
)


reranker_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a system that selects texts relevant to the query for further transmission of information to the final answer generation system.
You will be provided with a user QUERY and TEXT. Your task is to rate on a 10-point scale how useful the TEXT is
for answering the QUERY.

Format your response in the following JSON format:
{response_format_description}

Respond only in English and use JSON format.
"""),
        ("user", """
QUERY:
{question}

TEXT:
{context}
""")
    ]
)


chat_answer_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a system answering user questions based on sources. "
                   "You are provided with a user question and a set of sources. "
                   "For each source, a publication date may be specified. "
                   "Use it to indicate the exact year in the answer "
                   "if the source contains references to past or future years. "
                   "Use only information from the provided sources for the answer. "
                   "If the question cannot be answered based on the sources, "
                   "write \"Insufficient information to answer\"."),
        ("human", "{context}\nQuestion: {question}\n")
    ]
)

