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
2) subject - subject of the question that points to the target entity
3) target - entity which is the answer to edit_prompt
4) generalization - contains:
   - generalization_prompt: a reformulated edit_prompt to measure the success rate of editing within the editing scope
   - generalization_answer: the expected answer to the generalization prompt
5) locality - contains:
   - locality_prompt: prompt-question that checks whether the model's output changes after editing for unrelated inputs
   - locality_answer: the expected answer to the locality prompt
6) portability - contains:
   - portability_prompt: prompt-question that helps to measure the success rate of editing for reasoning/application
   - portability_answer: the expected answer to the portability prompt
7) rephrase - alternative ways to phrase the edit prompt

IMPORTANT: Your response must be a valid JSON object. Do not include any text before or after the JSON object.
The JSON object must start with {{ and end with }}. All strings must be properly quoted.

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
    "generalization": {{
        "generalization_prompt": "What is the name of the current President of the United States?",
        "generalization_answer": "Donald Trump"
    }},
    "locality": {{
        "locality_prompt": "Where is the capital of the United States?",
        "locality_answer": "Washington, D.C."
    }},
    "portability": {{
        "portability_prompt": "Where is the current U.S. President born?",
        "portability_answer": "Queens, New York City, New York"
    }},
    "rephrase": [
        "Name the current President of the United States",
        "Who currently holds the office of President in the United States?",
        "What is the name of the person currently serving as U.S. President?"
    ]
}}

# User input
ENTITIES: {entities}
RELATIONSHIP: {relationships}
DESCRIPTION: {descriptions}

# Output
Strictly use the same language as in the user input. Your answer must be a valid JSON object that strictly follows this schema:
{schema}

Remember: Your response must be a valid JSON object starting with {{ and ending with }}. Do not include any text before or after the JSON object.
"""

PREPARE_FOR_KNOWLEDGE_EDITING_PROMPT_WIKIDATA_RECENT_TYPE = """
You are an AI assistant that helps a human in preparing input data for knowledge editing.

# Goal
Your goal is to make input examples that will be used for editing and checking editing success.

# Description
Your goal is to make input examples that will be used for editing and checking editing success.
From ENTITIES (main entities in the text), RELATIONSHIP between entities and DESCRIPTION of the entities prepare: 
1) prompt - An input relationship converted to a question where answer is one of the entities (so called target entity). Note that subject should be included in the prompt.
2) subject - subject of the question, which is one of the entities, that points to the target entity
3) target_new - entity which is the answer to the prompt
4) portability - contains:
   - Logical_Generalization - a way to measure that model can use the edited knowledge when we slightly change the question. Contains:
      - prompt - question that helps to measure the success rate of editing for reasoning/application. That is some reasoning over prompt 
      - ground_truth - list of ground truth answers to the prompt - answer variants that should be considered as correct
   - Reasoning - a way to measure that model can use the edited knowledge for reasoning over the characteristics of target entity. Contains:
      - prompt - question that helps to measure the success rate of editing for reasoning/application. That is some reasoning over prompt 
      - ground_truth - list of ground truth answers to the prompt - answer variants that should be considered as correct
   - Subject_Aliasing - a way to measure that model can still name correct target entity when we rephrase the prompt. Contains:
      - prompt - question that helps to measure the success rate of editing for subject aliasing
      - ground_truth - list of ground truth answers to the prompt - answer variants that should be considered as correct
5) locality - contains:
   - Relation_Specificity - checking the knowledge on subject related questions (can be derived from entity description). Contains:
      - prompt - question on the subject.
      - ground_truth - list of ground truth answers to the prompt - answer variants that should be considered as correct
      
# Example
## Example Input 
ENTITIES: Leo Arons, Berlin
RELATIONSHIP: The place of death of Leo Arons is Berlin.
DESCRIPTION: Leo Arons, an experimental physicist, whose father was Albert Arons, was born in Vienna, Austria on 1890-01-01 and died in Berlin, Germany on 1945-01-01. Berlin is the capital of Germany with government headed by Kai Peter Wegner.

## Example Output
{{
    "subject": "Leo Arons",
    "prompt": "The place of death of Leo Arons is",
    "target_new": "Berlin",
    "portability": {{
            "Logical_Generalization": [
                {{
                    "prompt": "Is Leo Arons still alive?",
                    "ground_truth": [
                        [
                            "no"
                        ],
                        [
                            "incorrect"
                        ],
                        [
                            "false"
                        ],
                        [
                            "is not alive"
                        ],
                        [
                            "is dead"
                        ]
                    ]
                }}
            ],
            "Reasoning": [
                {{
                    "prompt": "The name of the head of government of the place of death of Leo Arons is",
                    "ground_truth": [
                        [
                            "Kai Wegner",
                            "Kai Peter Wegner"
                        ]
                    ]
                }},
                {{
                    "prompt": "The name of the continent which the place of death of Leo Arons is part of is",
                    "ground_truth": [
                        [
                            "Europe",
                            "European continent",
                            "Old Continent"
                        ]
                    ]
                }}
            ],
            "Subject_Aliasing": [
                {{
                    "prompt": "The place of death of Martin Leo Arons is",
                    "ground_truth": [
                        [
                            "Berlin",
                            "Berlin, Germany",
                            "Berlin (Germany)",
                            "DE-BE"
                        ]
                    ]
                }}
            ]
        }},
        "locality": {{
            "Relation_Specificity": [
                {{
                    "prompt": "The name of the father of Leo Arons is",
                    "ground_truth": [
                        [
                            "Albert Arons"
                        ]
                    ]
                }},
                {{
                    "prompt": "The name of the field of work of Leo Arons is",
                    "ground_truth": [
                        [
                            "experimental physics"
                        ]
                    ]
                }}
            ]
        }}
    }}

# User input
ENTITIES: {entities}
RELATIONSHIP: {relationships}
DESCRIPTION: {descriptions}
    
# Your answer
Strictly use the same language as in the user input. Your answer must be a valid JSON object that strictly follows this schema:
{schema}
    
"""