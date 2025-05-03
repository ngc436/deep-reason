#!/bin/bash

# Script to export data from Elasticsearch to CSV
# Usage: ./es_to_csv.sh -i <index> [-h <host>] [-p <port>] [-u <user>] [-P <password>] [-o <output_file>]

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo "Export data from Elasticsearch to CSV"
    echo ""
    echo "Options:"
    echo "  -i, --index       Elasticsearch index name (required)"
    echo "  -h, --host        Elasticsearch host (default: localhost)"
    echo "  -p, --port        Elasticsearch port (default: 9200)"
    echo "  -u, --user        Elasticsearch username"
    echo "  -P, --password    Elasticsearch password"
    echo "  -o, --output      Output CSV file path (default: output.csv)"
    echo "  --help            Show this help message"
    exit 0
}

# Default values
ES_HOST="localhost"
ES_PORT="9200"
OUTPUT_FILE="output.csv"
CURL_TIMEOUT=30  # Timeout in seconds

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--index)
            ES_INDEX="$2"
            shift 2
            ;;
        -h|--host)
            ES_HOST="$2"
            shift 2
            ;;
        -p|--port)
            ES_PORT="$2"
            shift 2
            ;;
        -u|--user)
            ES_USER="$2"
            shift 2
            ;;
        -P|--password)
            ES_PASS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check required parameters
if [ -z "$ES_INDEX" ]; then
    echo "Error: Index name is required"
    show_help
    exit 1
fi

# Construct curl command with authentication if provided
CURL_CMD="curl --connect-timeout $CURL_TIMEOUT --max-time $CURL_TIMEOUT"
if [ ! -z "$ES_USER" ] && [ ! -z "$ES_PASS" ]; then
    CURL_CMD="$CURL_CMD -u $ES_USER:$ES_PASS"
fi

# Function to check Elasticsearch connection
check_connection() {
    echo "Checking connection to Elasticsearch at http://$ES_HOST:$ES_PORT..."
    local response=$($CURL_CMD -s -X GET "http://$ES_HOST:$ES_PORT")
    local exit_code=$?
    
    if [ $exit_code -eq 28 ]; then
        echo "Error: Connection timed out after $CURL_TIMEOUT seconds"
        echo "Please check if Elasticsearch is running and accessible"
        exit 1
    elif [ $exit_code -ne 0 ]; then
        echo "Error: Cannot connect to Elasticsearch (exit code: $exit_code)"
        echo "Please check if Elasticsearch is running and accessible"
        exit 1
    fi
    
    if [ -z "$response" ]; then
        echo "Error: Empty response from Elasticsearch"
        exit 1
    fi
    
    echo "Connection successful"
}

# Function to list all available indices
list_indices() {
    echo "Listing all available indices..."
    local response=$($CURL_CMD -s -X GET "http://$ES_HOST:$ES_PORT/_cat/indices?v&h=index,health,status,docs.count,store.size")
    local exit_code=$?
    
    if [ $exit_code -eq 28 ]; then
        echo "Error: Request timed out after $CURL_TIMEOUT seconds"
        exit 1
    elif [ $exit_code -ne 0 ]; then
        echo "Error: Failed to list indices (exit code: $exit_code)"
        exit 1
    fi
    
    if [ -z "$response" ]; then
        echo "No indices found in Elasticsearch"
        return 1
    fi
    
    echo "Available indices:"
    echo "$response" | awk 'NR>1 {print $1}' | sort
    return 0
}

# Function to check if index exists
check_index() {
    echo "Checking if index $ES_INDEX exists..."
    local response=$($CURL_CMD -s -I -X HEAD "http://$ES_HOST:$ES_PORT/$ES_INDEX")
    local exit_code=$?
    
    if [ $exit_code -eq 28 ]; then
        echo "Error: Request timed out after $CURL_TIMEOUT seconds"
        echo "Please check if Elasticsearch is running and accessible"
        exit 1
    elif [ $exit_code -ne 0 ]; then
        echo "Error: Failed to check index (exit code: $exit_code)"
        exit 1
    fi
    
    # Check if the response contains 200 OK
    if echo "$response" | grep -q "200 OK"; then
        echo "Index exists and is accessible"
        
        # Get index details
        local details=$($CURL_CMD -s -X GET "http://$ES_HOST:$ES_PORT/_cat/indices/$ES_INDEX?v&h=health,status,docs.count,store.size")
        if [ -n "$details" ]; then
            echo "Index details:"
            echo "$details" | awk 'NR>1'
        fi
    else
        echo "Error: Index $ES_INDEX does not exist"
        list_indices
        exit 1
    fi
}

# Function to get document count
get_doc_count() {
    echo "Getting document count..."
    local response=$($CURL_CMD -s -X GET "http://$ES_HOST:$ES_PORT/$ES_INDEX/_count")
    local exit_code=$?
    
    if [ $exit_code -eq 28 ]; then
        echo "Error: Request timed out after $CURL_TIMEOUT seconds"
        exit 1
    elif [ $exit_code -ne 0 ]; then
        echo "Error: Failed to get document count (exit code: $exit_code)"
        exit 1
    fi
    
    if [ -z "$response" ]; then
        echo "Error: Empty response from Elasticsearch"
        exit 1
    fi
    
    local count=$(echo "$response" | jq -r '.count // 0')
    echo "Found $count documents"
    return 0
}

# Function to get field names from the index
get_field_names() {
    echo "Getting field names from index..."
    local response=$($CURL_CMD -s -X GET "http://$ES_HOST:$ES_PORT/$ES_INDEX/_mapping")
    local exit_code=$?
    
    if [ $exit_code -eq 28 ]; then
        echo "Error: Request timed out after $CURL_TIMEOUT seconds"
        exit 1
    elif [ $exit_code -ne 0 ]; then
        echo "Error: Failed to get mapping (exit code: $exit_code)"
        exit 1
    fi
    
    if [ -z "$response" ]; then
        echo "Error: Could not get mapping for index $ES_INDEX"
        exit 1
    fi
    
    # Try different mapping structures
    local fields=$(echo "$response" | jq -r '.[].mappings.properties | keys[] // empty')
    if [ -z "$fields" ]; then
        fields=$(echo "$response" | jq -r '.[].mappings | keys[] // empty')
        if [ -z "$fields" ]; then
            echo "Warning: No fields found in index mapping"
            return 1
        fi
    fi
    
    echo "$fields"
    return 0
}

# Function to export data to CSV
export_to_csv() {
    # Get field names
    FIELDS=$(get_field_names)
    if [ $? -ne 0 ]; then
        echo "Error: Failed to get field names from index $ES_INDEX"
        echo "Trying to get sample document to determine fields..."
        
        # Try to get a sample document to determine fields
        local sample_response=$($CURL_CMD -s -X GET "http://$ES_HOST:$ES_PORT/$ES_INDEX/_search?size=1")
        if [ -n "$sample_response" ]; then
            FIELDS=$(echo "$sample_response" | jq -r '.hits.hits[0]._source | keys[] // empty' | tr '\n' ',' | sed 's/,$//')
            if [ -z "$FIELDS" ]; then
                echo "Error: Could not determine fields from sample document"
                exit 1
            fi
        else
            echo "Error: Could not get sample document"
            exit 1
        fi
    else
        FIELDS=$(echo "$FIELDS" | tr '\n' ',' | sed 's/,$//')
    fi
    
    echo "Fields found: $FIELDS"
    
    # Create CSV header
    echo "$FIELDS" > "$OUTPUT_FILE"
    
    # Get all documents and convert to CSV
    echo "Exporting documents to CSV..."
    local response=$($CURL_CMD -s -X GET "http://$ES_HOST:$ES_PORT/$ES_INDEX/_search?size=10000")
    local exit_code=$?
    
    if [ $exit_code -eq 28 ]; then
        echo "Error: Request timed out after $CURL_TIMEOUT seconds"
        exit 1
    elif [ $exit_code -ne 0 ]; then
        echo "Error: Failed to get documents (exit code: $exit_code)"
        exit 1
    fi
    
    if [ -z "$response" ]; then
        echo "Error: No response from Elasticsearch"
        exit 1
    fi
    
    local hits=$(echo "$response" | jq -r '.hits.hits | length // 0')
    if [ "$hits" -eq 0 ]; then
        echo "Warning: No documents found in index $ES_INDEX"
        return 0
    fi
    
    echo "$response" | jq -r ".hits.hits[]._source | [.[$FIELDS]] | @csv" >> "$OUTPUT_FILE"
    echo "Exported $hits documents to $OUTPUT_FILE"
    return 0
}

# Main execution
echo "Starting export process..."
check_connection
list_indices
check_index
get_doc_count
export_to_csv

if [ $? -eq 0 ]; then
    echo "Export completed successfully"
    echo "Output file: $OUTPUT_FILE"
    echo "Number of lines in output: $(wc -l < "$OUTPUT_FILE")"
else
    echo "Export failed"
    exit 1
fi 