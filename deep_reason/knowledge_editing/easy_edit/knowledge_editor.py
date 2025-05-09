import json
import logging
import gc
import os
import yaml
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor import (
    BaseEditor,
    ROMEHyperParams,
    MEMITHyperParams,
    IKEHyperParams,
    WISEHyperParams
)
import traceback
from difflib import SequenceMatcher

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

class KnowledgeEditorError(Exception):
    """Base exception for KnowledgeEditor errors."""
    pass

class ModelLoadingError(KnowledgeEditorError):
    """Raised when there's an error loading the model or tokenizer."""
    pass

class DatasetError(KnowledgeEditorError):
    """Raised when there's an error with the dataset."""
    pass

class EditingError(KnowledgeEditorError):
    """Raised when there's an error during the editing process."""
    pass

class DeviceError(KnowledgeEditorError):
    """Raised when there's an error with CUDA device management."""
    pass

def get_available_gpus() -> List[int]:
    """Get list of available GPU devices.
    
    Returns:
        List of available GPU device IDs.
    """
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def set_cuda_device(device_id: Optional[int] = None):
    """Set the CUDA device to use.
    
    Args:
        device_id: ID of the CUDA device to use. If None, uses the first available device.
        
    Raises:
        DeviceError: If the specified device is not available.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Using CPU.")
        return
    
    available_devices = get_available_gpus()
    if not available_devices:
        logger.warning("No CUDA devices available. Using CPU.")
        return
    
    if device_id is None:
        device_id = available_devices[0]
    
    if device_id not in available_devices:
        raise DeviceError(f"CUDA device {device_id} is not available. Available devices: {available_devices}")
    
    torch.cuda.set_device(device_id)
    logger.info(f"Using CUDA device: {device_id} ({torch.cuda.get_device_name(device_id)})")

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("GPU memory cleared")

class KnowledgeEditor:
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct", device_id: Optional[int] = None):
        """Initialize the KnowledgeEditor with a specific model.
        
        Args:
            model_name: Name of the model to use for editing.
            device_id: ID of the CUDA device to use. If None, uses the first available device.
            
        Raises:
            ModelLoadingError: If there's an error loading the model or tokenizer.
            DeviceError: If there's an error with CUDA device management.
        """
        self.model_name = model_name
        logger.info(f"Initializing KnowledgeEditor with model: {model_name}")
        
        # Set CUDA device
        set_cuda_device(device_id)
        
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer.
        
        Raises:
            ModelLoadingError: If there's an error loading the model or tokenizer.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            # Configure tokenizer padding
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = 'left'
            logger.info("Tokenizer loaded successfully")
            
            # Get current device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading model on device: {device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model or tokenizer: {str(e)}")
        
    def load_dataset(self, json_path: str) -> List[Dict[str, Any]]:
        """Load knowledge editing dataset from JSON file.
        
        Args:
            json_path: Path to the JSON dataset file.
            
        Returns:
            List of dataset items.
            
        Raises:
            DatasetError: If there's an error loading or validating the dataset.
        """
        logger.info(f"Loading dataset from: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DatasetError(f"Invalid JSON file: {str(e)}")
        except FileNotFoundError:
            raise DatasetError(f"Dataset file not found: {json_path}")
        except Exception as e:
            raise DatasetError(f"Error reading dataset: {str(e)}")
        
        self._validate_dataset(data)
        logger.info(f"Successfully loaded dataset with {len(data)} items")
        return data

    def _validate_dataset(self, data: List[Dict[str, Any]]):
        """Validate the dataset structure.
        
        Args:
            data: Dataset to validate.
            
        Raises:
            DatasetError: If the dataset structure is invalid.
        """
        if not isinstance(data, list):
            raise DatasetError("Dataset must be a list of items")
        
        for item in data:
            if "knowledge_editing_input" not in item:
                raise DatasetError("Each item must contain 'knowledge_editing_input'")
            
            for edit_input in item["knowledge_editing_input"]:
                required_fields = ["edit_prompt", "subject", "target", "rephrase"]
                missing_fields = [field for field in required_fields if field not in edit_input]
                if missing_fields:
                    raise DatasetError(f"Missing required fields in edit input: {missing_fields}")

    def prepare_edit_data(self, edit_input: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare edit data from knowledge_editing_input."""
        # Convert all string inputs to lowercase first
        edit_input = {k: v.lower() if isinstance(v, str) else v for k, v in edit_input.items()}
        
        logger.debug(f"Preparing edit data for prompt: {edit_input['edit_prompt']}")
        
        # Handle locality data
        locality = edit_input.get("locality", {})
        loc_prompt = ""
        if locality and isinstance(locality, dict):
            # Convert locality data to lowercase
            locality = {k: v.lower() if isinstance(v, str) else v for k, v in locality.items()}
            # Format locality data for WISE
            if "locality_prompt" in locality and "locality_answer" in locality:
                loc_prompt = locality["locality_prompt"]  # Remove "nq question:" prefix
                locality = [{
                    "prompt": locality["locality_prompt"],
                    "ground_truth": locality["locality_answer"]
                }]
        
        # Format rephrase prompts and convert to lowercase
        rephrase = edit_input.get("rephrase", [])
        if isinstance(rephrase, str):
            rephrase = [rephrase.lower()]
        elif isinstance(rephrase, list):
            rephrase = [p.lower() if isinstance(p, str) else p for p in rephrase]

        # Prepare the edit data with WISE-required fields
        edit_data = {
            "prompt": edit_input["edit_prompt"].lower(),
            "subject": edit_input["subject"].lower(),
            "target": edit_input["target"].lower(),  # Keep as 'target' to match input
            "rephrase": rephrase,  # Keep as 'rephrase' to match input
            "locality": locality,
            "loc_prompt": loc_prompt.lower(),  # Directly include without prefix
            "portability": {
                "prompt": edit_input.get("portability_prompt", "").lower()
            }
        }
        
        # Handle generalization data
        if "generalization" in edit_input:
            generalization = edit_input["generalization"]
            if isinstance(generalization, dict):
                generalization = {k: v.lower() if isinstance(v, str) else v for k, v in generalization.items()}
            edit_data["generalization"] = generalization
        
        logger.debug(f"Prepared edit data: {json.dumps(edit_data, indent=2)}")
        return edit_data

    def validate_edit_result(self, metrics: Dict[str, Any], method: str) -> bool:
        """Validate the quality of edit results.
        
        Args:
            metrics: Metrics from the edit operation.
            method: Name of the editing method.
            
        Returns:
            True if the edit was successful, False otherwise.
        """
        if not metrics:
            logger.warning(f"No metrics returned for {method}")
            return False
            
        # Handle nested metrics structure
        if isinstance(metrics, list) and len(metrics) > 0:
            metrics = metrics[0]  # Take first result if it's a list
            
        if 'post' in metrics:
            metrics = metrics['post']  # Get metrics from post-edit results
            
        # Extract rewrite accuracy
        rewrite_acc = None
        if 'rewrite_acc' in metrics:
            rewrite_acc = metrics['rewrite_acc']
            # Handle list values
            if isinstance(rewrite_acc, list):
                rewrite_acc = rewrite_acc[0]  # Take first value if it's a list
        elif 'rewrite' in metrics and 'acc' in metrics['rewrite']:
            rewrite_acc = metrics['rewrite']['acc']
            
        if rewrite_acc is None:
            logger.warning(f"Could not find rewrite accuracy in metrics for {method}")
            return False
            
        # Convert rewrite_acc to float if it's a string
        try:
            rewrite_acc = float(rewrite_acc)
        except (ValueError, TypeError):
            logger.warning(f"Invalid rewrite accuracy value for {method}: {rewrite_acc}")
            return False
            
        # Extract locality accuracy
        locality_acc = None
        if 'locality_acc' in metrics:
            locality_acc = metrics['locality_acc']
            # Handle list values
            if isinstance(locality_acc, list):
                locality_acc = locality_acc[0]  # Take first value if it's a list
        elif 'locality' in metrics:
            # Calculate average locality accuracy if it's a dictionary of accuracies
            acc_values = []
            for k, v in metrics['locality'].items():
                if k.endswith('_acc'):
                    try:
                        if isinstance(v, list):
                            v = v[0]  # Take first value if it's a list
                        acc_values.append(float(v))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid locality accuracy value for {method}: {v}")
                        continue
            if acc_values:
                locality_acc = sum(acc_values) / len(acc_values)
                
        if locality_acc is None:
            logger.warning(f"Could not find locality accuracy in metrics for {method}")
            return False
            
        # Convert locality_acc to float if it's a string
        try:
            locality_acc = float(locality_acc)
        except (ValueError, TypeError):
            logger.warning(f"Invalid locality accuracy value for {method}: {locality_acc}")
            return False
            
        # Quality thresholds
        if rewrite_acc < 0.5:
            logger.warning(f"Low rewrite accuracy for {method}: {rewrite_acc}")
            return False
            
        if locality_acc < 0.5:
            logger.warning(f"Low locality accuracy for {method}: {locality_acc}")
            return False
            
        return True

    def _generate_model_response(self, prompt: str, model=None) -> str:
        """Generate a response from the model for a given prompt.
        
        Args:
            prompt: The prompt to generate a response for.
            model: Optional model to use for generation. If None, uses self.model.
            
        Returns:
            The generated response.
        """
        try:
            model_to_use = model if model is not None else self.model
            
            # Format prompt with Question/Answer format if not already present
            if not prompt.startswith('Question:'):
                prompt = f'Question:{prompt} Answer:'
            
            # Tokenize with padding and max length
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                max_length=30,
                # truncation=True
            ).to(model_to_use.device)
            
            # Generate response
            outputs = model_to_use.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=15,
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
                # pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Get the max length of input tokens
            max_length = inputs['input_ids'].shape[-1]
            
            # Decode only the new tokens (after the input)
            response = self.tokenizer.decode(outputs[0][max_length:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""

    def _test_generalization(self, model, edited_model, generalization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test generalization with prompts from the JSON file.
        
        Args:
            model: Original model
            edited_model: Edited model
            generalization_data: Dictionary containing generalization prompts and answers
            
        Returns:
            Dictionary containing generalization test results
        """
        results = {
            "prompts": [],
            "pre_edit_outputs": [],
            "post_edit_outputs": [],
            "ground_truth": [],
            "generalization_accuracy": 0.0
        }
        
        try:
            # Extract generalization prompt and answer
            if not generalization_data:
                logger.warning("No generalization data provided")
                return results
            
            prompt = generalization_data.get("generalization_prompt", "").lower()
            ground_truth = generalization_data.get("generalization_answer", "").lower()
            
            if not prompt or not ground_truth:
                logger.warning("Missing generalization prompt or answer")
                return results
            
            # Format prompt with Question/Answer format if not already present
            if not prompt.startswith('question:'):
                prompt = f'question:{prompt} answer:'
            
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                max_length=30,
                # truncation=True
            ).to(model.device)
            
            # Generate responses from both models
            pre_edit_outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=15,
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
                # pad_token_id=self.tokenizer.eos_token_id
            )
            
            post_edit_outputs = edited_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=15,
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
                # pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Get the max length of input tokens
            max_length = inputs['input_ids'].shape[-1]
            
            # Decode and store results
            pre_edit_response = self.tokenizer.decode(pre_edit_outputs[0][max_length:], skip_special_tokens=True).strip().lower()
            post_edit_response = self.tokenizer.decode(post_edit_outputs[0][max_length:], skip_special_tokens=True).strip().lower()
            
            results["prompts"].append(prompt)
            results["pre_edit_outputs"].append(pre_edit_response)
            results["post_edit_outputs"].append(post_edit_response)
            results["ground_truth"].append(ground_truth)
            
            # Log the results
            logger.info("Generalization Test:")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Pre-Edit Output: {pre_edit_response}")
            logger.info(f"Post-Edit Output: {post_edit_response}")
            logger.info(f"Ground Truth: {ground_truth}")
            logger.info("-" * 50)
            
            # Calculate generalization accuracy by comparing post-edit output with ground truth
            similarity = SequenceMatcher(None, post_edit_response, ground_truth).ratio()
            results["generalization_accuracy"] = similarity
            
        except Exception as e:
            logger.error(f"Error in generalization testing: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            results["error"] = str(e)
        
        return results

    def calculate_quality_metrics(self, metrics: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Calculate detailed quality metrics for knowledge editing results.
        
        Args:
            metrics: Metrics from the edit operation.
            method: Name of the editing method.
            
        Returns:
            Dictionary containing quality metrics.
        """
        quality_metrics = {
            "method": method,
            "success": False,
            "rewrite_accuracy": 0.0,
            "locality_accuracy": 0.0,
            "portability_accuracy": 0.0,
            "generalization_accuracy": 0.0,
            "fluency_score": 0.0,
            "edit_time": 0.0,
            "error_message": None,
            "generalization_test_results": None
        }
        
        try:
            # Handle nested metrics structure
            if isinstance(metrics, list) and len(metrics) > 0:
                metrics = metrics[0]  # Take first result if it's a list
                
            if 'post' in metrics:
                post_metrics = metrics['post']
            else:
                post_metrics = metrics
                
            # Extract rewrite accuracy
            if 'rewrite_acc' in post_metrics:
                rewrite_acc = post_metrics['rewrite_acc']
                if isinstance(rewrite_acc, list):
                    rewrite_acc = rewrite_acc[0]
                quality_metrics['rewrite_accuracy'] = float(rewrite_acc)
                
            # Extract locality accuracy
            if 'locality' in post_metrics:
                locality_accs = []
                for k, v in post_metrics['locality'].items():
                    if k.endswith('_acc'):
                        if isinstance(v, list):
                            v = v[0]
                        try:
                            locality_accs.append(float(v))
                        except (ValueError, TypeError):
                            continue
                if locality_accs:
                    quality_metrics['locality_accuracy'] = sum(locality_accs) / len(locality_accs)
                    
            # Extract portability accuracy
            if 'portability' in post_metrics:
                portability_accs = []
                for k, v in post_metrics['portability'].items():
                    if k.endswith('_acc'):
                        if isinstance(v, list):
                            v = v[0]
                        try:
                            portability_accs.append(float(v))
                        except (ValueError, TypeError):
                            continue
                if portability_accs:
                    quality_metrics['portability_accuracy'] = sum(portability_accs) / len(portability_accs)
                    
            # Run generalization tests with prompts from the edit data
            if 'edited_model' in metrics and 'model' in metrics and 'generalization' in metrics:
                generalization_results = self._test_generalization(
                    metrics['model'],
                    metrics['edited_model'],
                    metrics['generalization']
                )
                quality_metrics['generalization_test_results'] = generalization_results
                quality_metrics['generalization_accuracy'] = generalization_results['generalization_accuracy']
                
            # Calculate overall success
            quality_metrics['success'] = (
                quality_metrics['rewrite_accuracy'] >= 0.5 and
                quality_metrics['locality_accuracy'] >= 0.5 and
                quality_metrics['generalization_accuracy'] >= 0.5
            )
            
        except Exception as e:
            quality_metrics['error_message'] = str(e)
            logger.error(f"Error calculating quality metrics: {str(e)}")
            
        return quality_metrics

    def _run_edit_method(self, method_name: str, hparams_path: str, edit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific editing method.
        
        Args:
            method_name: Name of the editing method.
            hparams_path: Path to the hyperparameters file.
            edit_data: Data for the edit operation.
            
        Returns:
            Dictionary containing the edit results.
            
        Raises:
            EditingError: If there's an error during the editing process.
        """
        logger.info(f"Running {method_name} edit for prompt: {edit_data['prompt']}")
        
        try:
            # Generate pre-edit response
            pre_edit_response = self._generate_model_response(edit_data['prompt'])
            logger.info(f"Pre-edit response: {pre_edit_response}")
            
            # Debug: Print input data
            logger.debug(f"Edit data: {json.dumps(edit_data, indent=2)}")
            
            hparams_class = {
                "ROME": ROMEHyperParams,
                "MEMIT": MEMITHyperParams,
                "IKE": IKEHyperParams,
                "WISE": WISEHyperParams
            }[method_name]
            
            # Debug: Check if hyperparameters file exists
            if not os.path.exists(hparams_path):
                raise EditingError(f"Hyperparameters file not found: {hparams_path}")
            
            try:
                # Load YAML file
                with open(hparams_path, 'r') as f:
                    hparams_dict = yaml.safe_load(f)
                logger.debug(f"Loaded hyperparameters dict: {hparams_dict}")
                
                # Create hyperparameters object
                hparams = hparams_class(**hparams_dict)
                logger.debug(f"Created hyperparameters object: {hparams}")
            except Exception as e:
                logger.error(f"Error loading hyperparameters: {str(e)}")
                raise
            
            try:
                editor = BaseEditor.from_hparams(hparams)
                logger.debug("BaseEditor initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing BaseEditor: {str(e)}")
                raise
            
            # Ensure we're using the correct device
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                logger.debug(f"Using CUDA device {current_device} for {method_name} edit")
            
            # Debug: Print edit parameters
            logger.debug(f"Edit parameters:")
            logger.debug(f"  prompts: {edit_data['prompt']}")
            logger.debug(f"  target: {edit_data['target']}")
            logger.debug(f"  subject: {edit_data['subject']}")
            logger.debug(f"  rephrase: {edit_data['rephrase']}")
            logger.debug(f"  loc_prompt: {edit_data.get('loc_prompt', '')}")
            
            try:
                # Convert single values to lists for WISE
                if method_name == "WISE":
                    # Store original prompt for post-edit response
                    original_prompt = edit_data["prompt"]
                    
                    # Format data for WISE
                    prompts = [f"question:{edit_data['prompt']} answer:"]
                    target_new = [edit_data["target"]]  # Use 'target' instead of 'target_new'
                    subject = edit_data["subject"]
                    
                    # Handle rephrase prompts - ensure it's a list of lists
                    rephrase = edit_data.get("rephrase", [])
                    if isinstance(rephrase, str):
                        rephrase = [rephrase]
                    if not rephrase:
                        rephrase = [[""]]  # Provide empty list as fallback
                    elif isinstance(rephrase[0], str):
                        # Format each rephrase prompt
                        formatted_rephrase = [f"question:{p} answer:" for p in rephrase]
                        rephrase = [formatted_rephrase]  # Wrap in another list
                    
                    # Use provided locality prompts
                    loc_prompts = [
                        "nq question: ek veer ki ardaas veera meaning in english a brother's prayer... veera",
                        "nq question: where are the winter olympics going to be seoul",
                        "nq question: physician who studies and treats diseases of the endocrine system endocrinologist"
                    ]
                    
                    # Handle locality data
                    locality = edit_data.get("locality", [])
                    if isinstance(locality, dict):
                        locality = [locality]
                    
                    # Handle portability data
                    portability = edit_data.get("portability", {})
                    if isinstance(portability, dict):
                        portability = [portability]

                    # Handle generalization data
                    generalization = edit_data.get("generalization", {})
                    if isinstance(generalization, dict):
                        generalization = [generalization]
                    
                    # Create the request dictionary
                    request = {
                        "prompt": prompts[0],
                        "target_new": target_new[0],
                        "subject": subject,
                        "rephrase_prompts": rephrase[0],  # Use rephrase for WISE
                        "loc_prompt": loc_prompts[0],  # Use first locality prompt
                        "locality": locality[0] if locality else {},
                        "portability": portability[0] if portability else {},
                        "generalization": generalization[0] if generalization else {}
                    }
                    
                    # Call editor.edit with the correct parameters
                    metrics, edited_model, _ = editor.edit(
                        prompts=prompts,
                        target_new=target_new,
                        subject=subject,
                        rephrase_prompts=rephrase,  # Use rephrase for WISE
                        loc_prompts=loc_prompts[0],  # Use first locality prompt
                        locality=locality,
                        portability=portability,
                        generalization=generalization,
                        sequential_edit=True,
                        verbose=True
                    )
                else:
                    metrics, edited_model, _ = editor.edit(
                        prompts=edit_data["prompt"],
                        target_new=edit_data["target"],  # Use 'target' instead of 'target_new'
                        subject=edit_data["subject"],
                        rephrase_prompts=edit_data["rephrase"],  # Use 'rephrase' instead of 'rephrase_prompts'
                        generalization=edit_data.get("generalization", {}),
                        sequential_edit=True
                    )
                logger.debug(f"Edit completed with metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error during edit operation: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                raise
            
            # Generate post-edit response using the edited model
            post_edit_prompt = original_prompt if method_name == "WISE" else edit_data["prompt"]
            post_edit_response = self._generate_model_response(post_edit_prompt, model=edited_model)
            logger.info(f"Post-edit response: {post_edit_response}")
            
            # Create a dictionary to store metrics and models for quality calculation
            metrics_dict = {
                "metrics": metrics[0] if isinstance(metrics, list) else metrics,
                "model": self.model,
                "edited_model": edited_model
            }
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(metrics_dict, method_name)
            logger.info(f"Quality metrics: {json.dumps(quality_metrics, indent=2)}")
            
            # Clean up
            del edited_model
            clear_gpu_memory()
            
            return {
                "method": method_name,
                "metrics": metrics,
                "quality_metrics": quality_metrics,
                "pre_edit_response": pre_edit_response,
                "post_edit_response": post_edit_response
            }
        except Exception as e:
            logger.error(f"Error in {method_name} edit: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            clear_gpu_memory()
            raise EditingError(f"{method_name} edit failed: {str(e)}")

    def run_rome_edit(self, edit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run ROME (Rank-One Model Editing) method."""
        return self._run_edit_method("ROME", "hparams/ROME/llama3-8b.yaml", edit_data)

    def run_memit_edit(self, edit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run MEMIT (Memory Editing) method."""
        return self._run_edit_method("MEMIT", "hparams/MEMIT/llama-7b.yaml", edit_data)

    def run_ike_edit(self, edit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run IKE (In-Context Knowledge Editing) method."""
        return self._run_edit_method("IKE", "hparams/IKE/llama3-8b.yaml", edit_data)

    def run_wise_edit(self, edit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run WISE (Weight-Informed Selective Editing) method."""
        return self._run_edit_method("WISE", "hparams/WISE/llama3-8b.yaml", edit_data) 