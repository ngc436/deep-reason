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
You are an AI assistant that helps a human in preparing input data for knowledge editing.

# Goal
Your goal is to make input examples that will be used for editing and checking editing success.
From ENTITIES (main entities in the text), RELATIONSHIP between entities and DESCRIPTION of the entities prepare: 
1) edit_prompt - an input relationship converted to a question where answer is one of the entities
2) subject - subject of the question
3) target - entity which is the answer to edit_prompt
4) generalization_prompt - prompt-question that is a reformulated edit_prompt to measure the success rate of editing withing the editing scope
5) locality_prompt - prompt-question that checks whether the model's output changes after editing for unrelated inputs. Here you should come up with some general knowledge connected to the edit_prompt
6) portability_prompt - prompt-question that helps to measure the success rate of editing for reasoning/application(one hop, synonym, logical generalization)
Look carefully at provided example below that provide understanding of how the input should look like.

# Example
## Input
ENTITIES: Donald Trump, United States
RELATIONSHIP: The current President of the United States is Donald Trump.
DESCRIPTION: Donald Trump, was born on June 14, 1946, in Queens, New York City, New York. The capital of the United States is Washington, D.C. (short for District of Columbia).
## Output
{{
	"edit_prompt": "Who is the current President of the United States?",
	"subject": "President",
	"target": "Donald Trump",
	"generalization_prompt": "What is the name of the current President of the United States?", 
	"locality_prompt": "Where is the capital of the United States?",
	"portability_prompt": "Where is the current U.S. President born?"
}}

# User input
ENTITIES: {entities}
RELATIONSHIP: {relationships}
DESCRIPTION: {descriptions}

# Output
Your answer must strictly follow this JSON schema:
{schema}
"""