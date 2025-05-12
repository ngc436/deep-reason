import json
import os
from typing import List, Dict, Any
from datetime import datetime

def convert_agent_results_to_editing_dataset(
    input_file: str,
    output_dir: str = "results",
    output_prefix: str = "knowledge_editing_dataset"
) -> str:
    """
    Convert agent results to a dataset format focused on knowledge editing inputs.
    
    Args:
        input_file: Path to the input JSON file containing agent results
        output_dir: Directory to save the converted dataset
        output_prefix: Prefix for the output filename
        
    Returns:
        Path to the created output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_file, 'r') as f:
        agent_results = json.load(f)
    
    # Convert to new format
    converted_data = []
    for result in agent_results:
        # Skip if no knowledge editing input
        if not result.get("knowledge_editing_input"):
            continue
            
        # Handle both single and list of knowledge editing inputs
        editing_inputs = result["knowledge_editing_input"]
        if not isinstance(editing_inputs, list):
            editing_inputs = [editing_inputs]
            
        for editing_input in editing_inputs:
            if not editing_input:
                continue
                
            # Create a copy of the original editing input
            observation = editing_input.copy()
            
            # Convert prompt to lowercase if it exists
            if "prompt" in observation:
                observation["prompt"] = observation["prompt"].lower()
            
            if "subject" in observation:
                observation["subject"] = observation["subject"].lower()
            
            # Skip if subject is not a substring of prompt
            if "subject" in observation and "prompt" in observation:
                if observation["subject"].lower() not in observation["prompt"].lower():
                    continue
            
            # Rename specific fields while preserving the original structure
            if "portability" in observation:
                portability = observation["portability"]
                if isinstance(portability, dict):
                    # Handle logical_generalization which is now a list of prompts
                    if "logical_generalization" in portability:
                        if isinstance(portability["logical_generalization"], list):
                            portability["Local_Generalization"] = portability.pop("logical_generalization")
                        else:
                            # If it's not a list, convert it to a list with a single item
                            portability["Local_Generalization"] = [portability.pop("logical_generalization")]
                    
                    if "reasoning" in portability:
                        portability["Reasoning"] = portability.pop("reasoning")
                    if "subject_aliasing" in portability:
                        portability["Subject_Aliasing"] = portability.pop("subject_aliasing")
            
            if "locality" in observation:
                locality = observation["locality"]
                if isinstance(locality, dict):
                    if "relation_specificity" in locality:
                        locality["Relation_Specificity"] = locality.pop("relation_specificity")
            
            converted_data.append(observation)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir,
        f"{output_prefix}_{timestamp}.json"
    )
    
    # Save converted data
    print(f"Saving converted data to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    return output_file

def convert_dataset_to_kblam_format(input_file: str, output_file: str):
    # name: name of entity, example "FERC"
    # description_type: the name of the property, example "purpose"
    # description: the value of the property, example "regulatory body"
    # Q: A question based on the triple, example "What is the purpose of FERC?"
    # A: An answer based on the triple, example "The purpose of FERC is regulatory body."
    # key_string: The key used in KBLaM (created with a template of "The {property name} of {entity name}")
    # example: "the purpose of FERC"
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    converted_data = []
    for result in data:
        observation = {
            "name": result["target_new"],
            "description_type": result["subject"],
            "description": result["description"],
            "Q": result["Q"],
            "A": result["A"],
            "key_string": result["key_string"]
        }
        pass
    
# [
#   {
#     "name": "Enron",
#     "description_type": "purpose",
#     "description": "promoted Dabhol Power Company",
#     "Q": "What is the purpose of Enron?",
#     "A": "The purpose of Enron is promoted Dabhol Power Company.",
#     "key_string": "the purpose of Enron"
#   },
#   {
#     "name": "Enron",
#     "description_type": "objectives",
#     "description": "result in fairer and more efficient markets",
#     "Q": "What is the objectives of Enron?",
#     "A": "The objectives of Enron is result in fairer and more efficient markets.",
#     "key_string": "the objectives of Enron"
#   },
#   {
#     "name": "Enron",
#     "description_type": "description",
#     "description": "organization that includes other Enron affiliates in contract definitions",
#     "Q": "What is the description of Enron?",
#     "A": "The description of Enron is organization that includes other Enron affiliates in contract definitions.",
#     "key_string": "the description of Enron"
#   },
#   {
#     "name": "FERC",
#     "description_type": "purpose",
#     "description": "regulatory body",
#     "Q": "What is the purpose of FERC?",
#     "A": "The purpose of FERC is regulatory body.",
#     "key_string": "the purpose of FERC"
#   }
# ]

    raise NotImplementedError("Not implemented yet")

convert_agent_results_to_editing_dataset('results/complex_agent_result_tat_personalii_2_community_chain4_20250512_161559.json')