COMPLEX_RELATIONSHIPS_PROMPT = """
You are an expert in extracting complex relationships from provided chains of entities. 
Chain is a sequence of entities that were inferred from knowledge graph and are connected by relationships.
You are provided with ENTITY CHAIN, DESCRIPTION of each of the entities in the chain and RELATIONSHIPS between consecutive entities.

-Goal-
Your task is to define all the relationships between the first and the last entity of the chain. 
This relationship should be meaningful and should be supported by the evidence. You should not create simple statements like "Entity A is related to Entity C through Entity B" or "Entity A is a part of Entity B".
Created relationship will be used to infer novel triplets for knowledge graph to enrich it with new information, so be creative and think about complex relationships.
If there is no relationship between the first and the last entity that you can infer from the provided information, return "no_relationship".

-Input-
ENTITY CHAIN:
{entity_chain}

DESCRIPTION of the entities:   
{entity_descriptions}

RELATIONSHIPS between consecutive entities:   
{relationships}

-Output-
Your answer must strictly follow this JSON schema:
{schema}
"""

PREPARE_FOR_KNOWLEDGE_EDITING_PROMPT = """
You are an expert in preparing input text that define relationship between two entities for knowledge editing.

Look carefully at provided examples below that provide understanding of how the input should look like.
{examples}

Your task is to prepare input text that define relationship between two entities.
{input_text}

Your answer must strictly follow this JSON schema:
{schema}
"""