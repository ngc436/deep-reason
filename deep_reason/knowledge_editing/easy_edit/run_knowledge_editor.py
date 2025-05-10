import json
import logging
import argparse
from typing import List, Dict, Any
import torch
from knowledge_editor import KnowledgeEditor, KnowledgeEditorError, get_available_gpus
import os

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
            edit_input = {k: v.lower() if isinstance(v, str) else v for k, v in edit_input.items()}
            logger.info(f"\nProcessing edit {i+1}/{len(data)}, input {j+1}/{len(item['knowledge_editing_input'])}")
            logger.info(f"Prompt: {edit_input['edit_prompt']}")
            logger.info(f"Target: {edit_input['target']}")
            logger.info(f"Subject: {edit_input['subject']}")

            generalization = edit_input.get("generalization", {})
            locality = edit_input.get("locality", {})
            
            # Prepare edit data with all required fields
            edit_data = {
                "prompt": edit_input["edit_prompt"],
                "subject": edit_input["subject"],
                "target": edit_input["target"],
                "rephrase": [i.lower() for i in edit_input["rephrase"]],
                "generalization": {k: v.lower() if isinstance(v, str) else v for k, v in generalization.items()},
                "locality": {k: v.lower() if isinstance(v, str) else v for k, v in locality.items()},
                "portability_prompt": edit_input.get("portability_prompt", "").lower()
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
                    
                    # Log rephrase responses
                    logger.info("\nRephrase Examples:")
                    logger.info("Pre-edit rephrase responses:")
                    for prompt, response in zip(edit_data['rephrase'], result.get('pre_edit_rephrase_responses', [])):
                        logger.info(f"Prompt: {prompt}")
                        logger.info(f"Response: {response}")
                    
                    logger.info("\nPost-edit rephrase responses:")
                    for prompt, response in zip(edit_data['rephrase'], result.get('post_edit_rephrase_responses', [])):
                        logger.info(f"Prompt: {prompt}")
                        logger.info(f"Response: {response}")
                    
                    # Log generalization responses
                    logger.info("\nGeneralization Examples:")
                    logger.info("Pre-edit generalization:")
                    logger.info(f"Prompt: {edit_data.get('generalization', {}).get('generalization_prompt', 'N/A')}")
                    logger.info(f"Response: {result.get('pre_edit_generalization', 'N/A')}")
                    
                    logger.info("\nPost-edit generalization:")
                    logger.info(f"Prompt: {edit_data.get('generalization', {}).get('generalization_prompt', 'N/A')}")
                    logger.info(f"Response: {result.get('post_edit_generalization', 'N/A')}")
                    
                    # Log portability responses
                    logger.info("\nPortability Examples:")
                    logger.info("Pre-edit portability:")
                    logger.info(f"Prompt: {edit_data.get('portability_prompt', 'N/A')}")
                    logger.info(f"Response: {result.get('pre_edit_portability', 'N/A')}")
                    
                    logger.info("\nPost-edit portability:")
                    logger.info(f"Prompt: {edit_data.get('portability_prompt', 'N/A')}")
                    logger.info(f"Response: {result.get('post_edit_portability', 'N/A')}")
                    
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

def save_results(results: List[Dict[str, Any]], output_path: str, method: str, dataset_name: str):
    """Save editing results to a JSON file.
    
    Args:
        results: List of edit results.
        output_path: Base path to save the results.
        method: Name of the editing method used.
        dataset_name: Name of the dataset used.
    """
    try:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Extract dataset name from path if not provided
        if not dataset_name:
            dataset_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Create filename with method and dataset name
        filename = f"results/{method}_{dataset_name}_results.json"
        
        # Save detailed results
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {filename}")
        
        # Save summary statistics
        stats = calculate_statistics(results)
        stats_filename = f"results/{method}_{dataset_name}_stats.json"
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_filename}")
        
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
    try:
        total_edits = len(results)
        successful_edits = 0
        
        # Count successful edits based on quality metrics
        for r in results:
            if "results" in r:
                for method_result in r["results"]:
                    if isinstance(method_result, dict):
                        # Check quality metrics for success
                        quality_metrics = method_result.get("quality_metrics", {})
                        if quality_metrics:
                            # Consider an edit successful if it meets minimum quality thresholds
                            success = (
                                quality_metrics.get("rewrite_accuracy", 0) >= 0.5 and
                                quality_metrics.get("locality_accuracy", 0) >= 0.5 and
                                quality_metrics.get("generalization_accuracy", 0) >= 0.5
                            )
                            if success:
                                successful_edits += 1
                                break
        
        # Calculate method-specific statistics
        method_stats = {}
        for method in ["ROME", "MEMIT", "IKE", "WISE"]:
            method_success = 0
            for r in results:
                if "results" in r:
                    for method_result in r["results"]:
                        if isinstance(method_result, dict) and method_result.get("method") == method:
                            quality_metrics = method_result.get("quality_metrics", {})
                            if quality_metrics:
                                success = (
                                    quality_metrics.get("rewrite_accuracy", 0) >= 0.5 and
                                    quality_metrics.get("locality_accuracy", 0) >= 0.5 and
                                    quality_metrics.get("generalization_accuracy", 0) >= 0.5
                                )
                                if success:
                                    method_success += 1
                                    break
            
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
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        # Return basic statistics in case of error
        return {
            "total_edits": len(results),
            "successful_edits": 0,
            "overall_success_rate": 0,
            "method_statistics": {},
            "error": str(e)
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
        
        # Get dataset name from path
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        
        # Process each method separately
        for method in args.methods:
            logger.info(f"\nProcessing edits using {method} method...")
            results = process_edits(editor, data, [method])  # Process one method at a time
            
            # Calculate and log statistics
            stats = calculate_statistics(results)
            logger.info(f"\n{method} Editing Statistics:")
            logger.info(f"Total edits: {stats['total_edits']}")
            logger.info(f"Successful edits: {stats['successful_edits']}")
            logger.info(f"Overall success rate: {stats['overall_success_rate']:.2%}")
            
            # Save results for this method
            save_results(results, args.output, method, dataset_name)
            
    except KnowledgeEditorError as e:
        logger.error(f"Knowledge editing error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 