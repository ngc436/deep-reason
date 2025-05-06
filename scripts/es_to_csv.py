import argparse
import csv
from elasticsearch import Elasticsearch, NotFoundError
from typing import List, Dict, Any
import sys
from datetime import datetime

def get_es_client(host: str, username: str, password: str) -> Elasticsearch:
    """Create and return an Elasticsearch client."""
    return Elasticsearch(
        hosts=host,
        basic_auth=(username, password),
        request_timeout=300000
    )

def list_indices(es_client: Elasticsearch) -> Dict[str, str]:
    """List all available indices in the Elasticsearch cluster with their creation times."""
    try:
        indices = es_client.indices.get(index='*')
        index_info = {}
        for index_name in indices.keys():
            settings = es_client.indices.get_settings(index=index_name)
            creation_date = settings[index_name]['settings']['index']['creation_date']
            # Convert Unix timestamp to human-readable date
            creation_time = datetime.fromtimestamp(int(creation_date)/1000).strftime('%Y-%m-%d %H:%M:%S')
            index_info[index_name] = creation_time
        return index_info
    except Exception as e:
        print(f"Error listing indices: {e}")
        sys.exit(1)

def get_index_fields(es_client: Elasticsearch, index_name: str) -> List[str]:
    """Retrieve all fields from an Elasticsearch index."""
    try:
        mappings = es_client.indices.get_mapping(index=index_name)
        index_mapping = mappings.get(index_name, {}).get('mappings', {})
        if 'properties' not in index_mapping:
            return []
        properties = index_mapping['properties']
        
        def extract_fields(props: Dict[str, Any], prefix: str = '') -> set:
            fields = set()
            for field, config in props.items():
                full_path = f"{prefix}{field}"
                fields.add(full_path)
                
                if 'properties' in config:
                    fields.update(extract_fields(config['properties'], f"{full_path}."))
                
                if 'fields' in config:
                    for sub_field, sub_config in config['fields'].items():
                        sub_path = f"{full_path}.{sub_field}"
                        fields.add(sub_path)
                        if 'properties' in sub_config:
                            fields.update(extract_fields(sub_config['properties'], f"{sub_path}."))
            return fields
        
        all_fields = extract_fields(properties)
        return sorted(all_fields)
    except Exception as e:
        print(f"Error getting index fields: {e}")
        sys.exit(1)

def get_documents(es_client: Elasticsearch, index_name: str) -> List[Dict[str, Any]]:
    """Fetch all documents from an Elasticsearch index using scroll API."""
    try:
        # Initialize scroll
        response = es_client.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            size=1000,  # Process in batches of 1000
            scroll='5m',  # Keep the scroll context alive for 5 minutes
            _source=True
        )
        
        scroll_id = response['_scroll_id']
        documents = [hit["_source"] for hit in response['hits']['hits']]
        total_docs = len(documents)
        
        print(f"Fetching documents... (found {total_docs} so far)", end='\r')
        
        # Continue scrolling until no more documents
        while True:
            response = es_client.scroll(
                scroll_id=scroll_id,
                scroll='5m'
            )
            
            hits = response['hits']['hits']
            if not hits:
                break
                
            new_docs = [hit["_source"] for hit in hits]
            documents.extend(new_docs)
            total_docs = len(documents)
            print(f"Fetching documents... (found {total_docs} so far)", end='\r')
        
        # Clear scroll context
        es_client.clear_scroll(scroll_id=scroll_id)
        print(f"\nSuccessfully fetched {total_docs} documents")
        
        return documents
    except NotFoundError:
        print(f"Index '{index_name}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching documents: {e}")
        sys.exit(1)

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def export_to_csv(documents: List[Dict[str, Any]], fields: List[str], output_file: str):
    """Export documents to CSV file with metadata as separate columns and remove empty columns."""
    try:
        # Flatten all documents and collect all possible fields
        flattened_docs = [flatten_dict(doc) for doc in documents]
        all_fields = set()
        for doc in flattened_docs:
            all_fields.update(doc.keys())
        
        # Remove empty columns and exclude 'embedding' and 'vector' columns
        non_empty_fields = []
        for field in sorted(all_fields):
            if field not in ['embedding', 'vector'] and any(doc.get(field) for doc in flattened_docs):
                non_empty_fields.append(field)
        
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=non_empty_fields)
            writer.writeheader()
            for doc in flattened_docs:
                # Only include non-empty fields and exclude 'embedding' and 'vector'
                row = {k: v for k, v in doc.items() if k in non_empty_fields and k not in ['embedding', 'vector']}
                writer.writerow(row)
        
        print(f"Successfully exported {len(documents)} documents to {output_file}")
        print(f"Exported {len(non_empty_fields)} non-empty columns")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Export Elasticsearch index to CSV')
    parser.add_argument('--host', required=True, help='Elasticsearch host URL')
    parser.add_argument('--username', required=True, help='Elasticsearch username')
    parser.add_argument('--password', required=True, help='Elasticsearch password')
    parser.add_argument('--index', help='Index name to export (optional)')
    parser.add_argument('--output', default='output.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Create Elasticsearch client
    es_client = get_es_client(args.host, args.username, args.password)
    
    # List available indices
    indices = list_indices(es_client)
    print("\nAvailable indices (sorted by creation time, most recent first):")
    # Sort indices by creation time in descending order
    sorted_indices = sorted(indices.items(), key=lambda x: x[1], reverse=True)
    for idx, creation_time in sorted_indices:
        print(f"- {idx} (Created: {creation_time})")
    
    # If no index specified, exit
    if not args.index:
        print("\nPlease specify an index using --index to export data")
        sys.exit(0)
    
    # Get fields and documents
    fields = get_index_fields(es_client, args.index)
    documents = get_documents(es_client, args.index)
    
    # Export to CSV
    export_to_csv(documents, fields, args.output)

if __name__ == "__main__":
    main()