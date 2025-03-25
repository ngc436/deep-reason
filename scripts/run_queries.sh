#!/bin/bash

# Default paths
DEFAULT_JSON_FILE="datasets/ObliQA/ObliQA_train_filtered_doc1.json"
DEFAULT_OUTPUT_CSV="query.csv"

# Parse command line arguments
JSON_FILE=${1:-$DEFAULT_JSON_FILE}
OUTPUT_CSV=${2:-$DEFAULT_OUTPUT_CSV}

# Display information about the files being used
echo "Using JSON file: $JSON_FILE"
echo "Results will be saved to: $OUTPUT_CSV"

# Create or clear the output CSV file
echo "question_id,question,result" > "$OUTPUT_CSV"

# Extract all questions from the JSON file and process each one
jq -r '.Question | to_entries[] | [.key, .value] | @tsv' "$JSON_FILE" | while IFS=$'\t' read -r id question; do
    echo "Processing question ID: $id"
    echo "Question: $question"
    
    # Run graphrag query with the current question
    result=$(graphrag query \
        --root /data \
        --method local \
        --query "$question")
    
    # Escape any commas and quotes in the result to ensure CSV validity
    escaped_result=$(echo "$result" | sed 's/"/""/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    escaped_question=$(echo "$question" | sed 's/"/""/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    
    # Append the result to the CSV file
    echo "\"$id\",\"$escaped_question\",\"$escaped_result\"" >> "$OUTPUT_CSV"
    
    echo "Result saved for question ID: $id"
    echo "---------------------------------"
done

echo "All queries processed. Results saved to $OUTPUT_CSV"

# Usage information
cat << EOF

Usage:
  ./$(basename "$0") [JSON_FILE] [OUTPUT_CSV]

Examples:
  ./$(basename "$0")                                               # Use default paths
  ./$(basename "$0") datasets/ObliQA/ObliQA_train_filtered_doc1.json query.csv  # Specify both paths
  ./$(basename "$0") custom_questions.json                         # Specify only JSON file, use default CSV
EOF