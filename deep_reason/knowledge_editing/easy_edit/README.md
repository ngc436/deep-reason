# Knowledge Editing with EasyEdit

This directory contains code for performing knowledge editing on a dataset using different methods from the EasyEdit library with the Meta-Llama-3-8B-Instruct model.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have access to the Llama-3-8B-Instruct model. You'll need to:
   - Have a Hugging Face account
   - Request access to the Llama-3 model
   - Set up your Hugging Face token
   - Accept the model's license agreement on Hugging Face

## Usage

The main script `run_knowledge_editing.py` implements three different knowledge editing methods:
- ROME (Rank-One Model Editing)
- MEMIT (Memory Editing)
- IKE (In-Context Knowledge Editing)

To run the knowledge editing:

```bash
python run_knowledge_editing.py
```

The script will:
1. Load the knowledge editing dataset from the specified JSON file
2. Process each knowledge editing input using different methods
3. Save the results to `editing_results.json`

## Input Data Format

The input JSON file should contain knowledge editing inputs in the following format:
```json
{
  "knowledge_editing_input": [
    {
      "edit_prompt": "Question about the knowledge to edit",
      "subject": "Subject of the edit",
      "target": "Target new knowledge",
      "rephrase": ["Rephrased prompts for evaluation"]
    }
  ]
}
```

## Output Format

The results are saved in `editing_results.json` with the following structure:
```json
[
  {
    "edit_prompt": "Original edit prompt",
    "results": [
      {
        "method": "ROME/MEMIT/IKE",
        "metrics": {
          "rewrite_acc": 0.8,
          // other metrics
        },
        "success": true
      }
    ]
  }
]
```

## Customization

You can modify the code to:
- Use different models by changing the `model_name` parameter in `KnowledgeEditor`
- Add more editing methods by implementing new methods in the `KnowledgeEditor` class
- Change the evaluation metrics and success criteria
- Modify the input/output file paths

## Notes on Llama-3

The code uses the Meta-Llama-3-8B-Instruct model, which requires:
- Trust remote code enabled for proper model loading
- Updated hyperparameters for each editing method
- Sufficient GPU memory (recommended 16GB+) 