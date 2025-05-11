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

convert_agent_results_to_editing_dataset('results/complex_agent_result_obliqa-full_community_chain3_20250511_155757.json')