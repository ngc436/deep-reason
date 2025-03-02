UNKNOWN_TERMS_PROMPT = """
# Instruction on how to handle unknown terms

If you see unknown terms, try to find the meaning of them in the context.
"""

# TODO: rewrite prompt 
KG_PROMPT_VAR1 = """
# Instruction for Creating Nodes and Triplets on a text fragment: 
Use a structured triplet format to capture data, as follows: "subject, relation, object". 
For example, from "Albert Einstein, born in Germany, is known for developing the theory of relativity," extract "Albert Einstein, country of birth, Germany; Albert Einstein, developed, Theory of Relativity." 
Remember that you should break complex triplets like "John, position, engineer in Google" into simple triplets like "John, position, engineer", "John, work at, Google". 
Length of your triplet should not be more than 7 words. 
You should extract only concrete knowledges, any assumptions must be described as hypothesis. 
For example, from phrase "John have scored many points and potentially will be winner" you should extract "John, scored many, points; John, could be, winner" and should not extract "John, will be, winner". 
Remember that object and subject must be an atomary units while relation can be more complex and long. 
If observation states that you take item, the triplet shoud be: 'item, is in, inventory' and nothing else.

Do not miss important information. 
If observation is 'book involves story about knight, who needs to kill a dragon', triplets should be 'book, involves, knight', 'knight, needs to kill, dragon'. 
If observation involves some type of notes, do not forget to include triplets about entities this note includes. 
There could be connections between distinct parts of observations. 
For example if there is information in the beginning of the observation that you are in location, and in the end it states that there is an exit to the east, you should extract triplet: 'location, has exit, east'. Several triplets can be extracted, that contain information about the same node. For example 'kitchen, contains, apple', 'kitchen, contains, table', 'apple, is on, table'.

Do not miss this type of connections. 
Other examples of triplets: 'room z, contains, black locker'; 'room x, has exit, east', 'apple, is on, table', 'key, is in, locker', 'apple, to be, grilled', 'potato, to be, sliced', ’stove, used for, frying’, ’recipe, requires, green apple’, ’recipe, requires, potato’. Do not include triplets that state the current location of an agent like ’you, are in, location’. 
Do not use 'none' as one of the entities. 
If there is information that you read something, do not forget to incluse triplets that state that entity that you read contains information that you extract.

Remember that triplets must be extracted in format:
{response_format_description}
"""
# Example of triplets you have extracted before: {example} Observation: {observation}

ONTOLOGY_PROMPT = """
You are a brilliant ontology agent for combining ontology from extracted triplets.
Ontology nodes should depict entities or concepts, similar to Wikipedia high-level nodes. 
Look carefully at the provided triplets and existing so far ontology and decide the following:
1. If the provided triplet can be added to the existing ontology, add it to the ontology.
2. If the provided triplet does not fit into the existing ontology, create a new ontology node for it.

Current ontology: {current_ontology}

Your answer should be in the following format:
{response_format_description}
"""