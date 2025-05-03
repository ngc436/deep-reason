COMPLEX_RELATIONSHIPS_PROMPT = """
You are an expert in extracting complex relationships from provided chains of entities.
You are provided with DESCRIPTION of each of the entities in the chain and RELATIONSHIPS between consecutive entities.

Your task is to define relationship between the first and the last entity of the chain. 
If there is no relationship between the first and the last entity that you can infer from the provided information, return "no_relationship".

DESCRIPTION of the entities: {entity_descriptions}

RELATIONSHIPS between consecutive entities: {relationships}

Your answer should be in the following format:
{response_format_description}
"""