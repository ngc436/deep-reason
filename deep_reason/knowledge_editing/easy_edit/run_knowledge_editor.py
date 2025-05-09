import json
import logging
import argparse
from typing import List, Dict, Any
import torch
from knowledge_editor import KnowledgeEditor, KnowledgeEditorError, get_available_gpus

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('editing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_edits(editor: KnowledgeEditor, data: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    """Process all edits in the dataset.
    
    Args:
        editor: KnowledgeEditor instance.
        data: Dataset containing edits to process.
        methods: List of editing methods to use.
        
    Returns:
        List of results for each edit.
    """
    results = []
    total_edits = sum(len(item["knowledge_editing_input"]) for item in data)
    logger.info(f"Processing {total_edits} total edits using methods: {', '.join(methods)}")
    
    for i, item in enumerate(data):
        for j, edit_input in enumerate(item["knowledge_editing_input"]):
            logger.info(f"\nProcessing edit {i+1}/{len(data)}, input {j+1}/{len(item['knowledge_editing_input'])}")
            logger.info(f"Prompt: {edit_input['edit_prompt']}")
            logger.info(f"Target: {edit_input['target']}")
            logger.info(f"Subject: {edit_input['subject']}")
            
            # Prepare edit data with all required fields
            edit_data = {
                "prompt": edit_input["edit_prompt"],
                "subject": edit_input["subject"],
                "target": edit_input["target"],
                "rephrase": edit_input["rephrase"],
                "generalization": edit_input.get("generalization", {}),
                "locality": edit_input.get("locality", {}),
                "portability_prompt": edit_input.get("portability_prompt", "")
            }
            
            edit_results = []
            for method in methods:
                try:
                    if method == "ROME":
                        result = editor.run_rome_edit(edit_data)
                    elif method == "MEMIT":
                        result = editor.run_memit_edit(edit_data)
                    elif method == "IKE":
                        result = editor.run_ike_edit(edit_data)
                    elif method == "WISE":
                        result = editor.run_wise_edit(edit_data)
                    
                    # Log the responses
                    logger.info(f"\n{method} Results:")
                    logger.info(f"Pre-edit response: {result.get('pre_edit_response', 'N/A')}")
                    logger.info(f"Post-edit response: {result.get('post_edit_response', 'N/A')}")
                    logger.info(f"Success: {result.get('success', False)}")
                    
                    edit_results.append(result)
                except KnowledgeEditorError as e:
                    logger.error(f"Error in {method} edit: {str(e)}")
                    edit_results.append({
                        "method": method,
                        "error": str(e),
                        "success": False
                    })
            
            results.append({
                "edit_prompt": edit_data["prompt"],
                "results": edit_results
            })
    
    return results

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save editing results to a JSON file.
    
    Args:
        results: List of edit results.
        output_path: Path to save the results.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics from the editing results.
    
    Args:
        results: List of edit results.
        
    Returns:
        Dictionary containing statistics.
    """
    total_edits = len(results)
    successful_edits = sum(1 for r in results if any(m["success"] for m in r["results"]))
    
    method_stats = {}
    for method in ["ROME", "MEMIT", "IKE", "WISE"]:
        method_success = sum(1 for r in results if any(
            m["success"] and m["method"] == method for m in r["results"]
        ))
        method_stats[method] = {
            "total": total_edits,
            "successful": method_success,
            "success_rate": method_success / total_edits if total_edits > 0 else 0
        }
    
    return {
        "total_edits": total_edits,
        "successful_edits": successful_edits,
        "overall_success_rate": successful_edits / total_edits if total_edits > 0 else 0,
        "method_statistics": method_stats
    }

def print_gpu_info():
    """Print information about available GPUs."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available. Using CPU.")
        return
    
    available_devices = get_available_gpus()
    logger.info(f"Found {len(available_devices)} CUDA device(s):")
    for device_id in available_devices:
        device_name = torch.cuda.get_device_name(device_id)
        memory_total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3  # Convert to GB
        memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # Convert to GB
        memory_free = memory_total - memory_allocated
        logger.info(f"  Device {device_id}: {device_name}")
        logger.info(f"    Total Memory: {memory_total:.2f} GB")
        logger.info(f"    Allocated Memory: {memory_allocated:.2f} GB")
        logger.info(f"    Free Memory: {memory_free:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Run knowledge editing on a dataset")
    parser.add_argument("--dataset", type=str, required=True,
                      help="Path to the dataset JSON file")
    parser.add_argument("--output", type=str, default="user_dataset_train/editing_results.json",
                      help="Path to save the results")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8B-Instruct",
                      help="Model to use for editing")
    parser.add_argument("--device", type=int, default=None,
                      help="CUDA device ID to use (default: first available device)")
    parser.add_argument("--methods", type=str, nargs="+", 
                      choices=["ROME", "MEMIT", "IKE", "WISE"],
                      default=["ROME", "MEMIT", "IKE", "WISE"],
                      help="Editing methods to use (default: all methods)")
    args = parser.parse_args()
    
    try:
        # Print GPU information
        print_gpu_info()
        
        # Initialize editor
        editor = KnowledgeEditor(model_name=args.model, device_id=args.device)
        
        # Load and process dataset
        data = editor.load_dataset(args.dataset)
        results = process_edits(editor, data, args.methods)
        
        # Calculate and log statistics
        stats = calculate_statistics(results)
        logger.info("Editing Statistics:")
        logger.info(f"Total edits: {stats['total_edits']}")
        logger.info(f"Successful edits: {stats['successful_edits']}")
        logger.info(f"Overall success rate: {stats['overall_success_rate']:.2%}")
        
        for method, method_stats in stats["method_statistics"].items():
            if method in args.methods:  # Only show stats for selected methods
                logger.info(f"\n{method} Statistics:")
                logger.info(f"  Total: {method_stats['total']}")
                logger.info(f"  Successful: {method_stats['successful']}")
                logger.info(f"  Success rate: {method_stats['success_rate']:.2%}")
        
        # Save results
        save_results(results, args.output)
        
    except KnowledgeEditorError as e:
        logger.error(f"Knowledge editing error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 