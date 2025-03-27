#!/bin/bash

# Define color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
RESET='\033[0m'

# Default paths
DEFAULT_JSON_FILE="datasets/ObliQA/ObliQA_train_filtered_doc1.json"
DEFAULT_OUTPUT_CSV="query.csv"

# Check for required commands
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: Required command '$1' not found.${RESET}"
        case "$1" in
            jq)
                echo -e "${YELLOW}Please install jq to process JSON data:${RESET}"
                echo "  - On macOS: brew install jq"
                echo "  - On Ubuntu/Debian: sudo apt-get install jq"
                echo "  - On CentOS/RHEL: sudo yum install jq"
                echo "  - Or download from: https://stedolan.github.io/jq/download/"
                ;;
            graphrag)
                echo -e "${YELLOW}Please ensure graphrag is installed and in your PATH${RESET}"
                ;;
            *)
                echo -e "${YELLOW}Please install $1 to continue.${RESET}"
                ;;
        esac
        exit 1
    fi
}

# Check required dependencies
check_command "jq"
check_command "graphrag"

# Parse command line arguments
JSON_FILE=${1:-$DEFAULT_JSON_FILE}
OUTPUT_CSV=${2:-$DEFAULT_OUTPUT_CSV}

# Verify JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo -e "${RED}Error: JSON file '$JSON_FILE' not found.${RESET}"
    echo -e "${YELLOW}Please check the file path and try again.${RESET}"
    exit 1
fi

# Display information about the files being used
echo -e "${CYAN}${BOLD}=== Query Processing Setup ===${RESET}"
echo -e "${CYAN}Using JSON file:${RESET} $JSON_FILE"
echo -e "${CYAN}Results will be saved to:${RESET} $OUTPUT_CSV"
echo -e "${CYAN}${BOLD}===============================${RESET}\n"

# Create or clear the output CSV file
echo "question_id,question,result" > "$OUTPUT_CSV"

# Extract all questions from the JSON file and process each one
echo -e "${CYAN}${BOLD}=== Starting Query Processing ===${RESET}\n"

jq -r '.Question | to_entries[] | [.key, .value] | @tsv' "$JSON_FILE" | while IFS=$'\t' read -r id question; do
    echo -e "${MAGENTA}Processing question ID:${RESET} ${BOLD}$id${RESET}"
    echo -e "${BLUE}Question:${RESET} ${BLUE}$question${RESET}"
    
    # Run graphrag query with the current question
    echo -e "${YELLOW}Running query...${RESET}"
    result=$(graphrag query \
        --root /data \
        --method global \
        --query "$question")
    
    # Display the result in green
    echo -e "${GREEN}Result:${RESET} ${GREEN}$result${RESET}"
    
    # Escape any commas and quotes in the result to ensure CSV validity
    escaped_result=$(echo "$result" | sed 's/"/""/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    escaped_question=$(echo "$question" | sed 's/"/""/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    
    # Append the result to the CSV file
    echo "\"$id\",\"$escaped_question\",\"$escaped_result\"" >> "$OUTPUT_CSV"
    
    echo -e "${CYAN}Result saved for question ID:${RESET} ${BOLD}$id${RESET}"
    echo -e "${MAGENTA}${BOLD}---------------------------------${RESET}\n"
done

echo -e "${CYAN}${BOLD}=== Processing Complete ===${RESET}"
echo -e "${GREEN}${BOLD}All queries processed. Results saved to $OUTPUT_CSV${RESET}\n"

# Usage information
cat << EOF
${YELLOW}${BOLD}Usage:${RESET}
  ./$(basename "$0") [JSON_FILE] [OUTPUT_CSV]

${YELLOW}${BOLD}Examples:${RESET}
  ./$(basename "$0")                                               # Use default paths
  ./$(basename "$0") datasets/ObliQA/ObliQA_train_filtered_doc1.json query.csv  # Specify both paths
  ./$(basename "$0") custom_questions.json                         # Specify only JSON file, use default CSV
EOF