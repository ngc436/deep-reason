## Input

Expected input to kg pipeline is a list of Chunks with the following fields:

```json
{
    "document_id": "document id",
    "chapter_name": "name of the chapter",
    "order_id": "order id of the chunk in the document",
    "text": "chunk text",
}
```


## Usage

### Indexing a dataset (using Elasticsearch)
To create an index and upload a dataset to Elasticsearch, run:
```bash
drctl es upload --dataset=ObliQA --dataset-path="<path to dataset>" --es-index="<index-name>"
```

Currently, the only supported dataset is ObliQA.

### Deleting an index
To delete an index, run:
```bash
drctl es drop-index --index-name=<index-name>
```

### Asking a question

To ask a question, run:
```bash
drctl rag ask --question='<question>' --es-index="<index-name>"
```

You may put all questions in a file (questions in JSON format [{'question': '...'}, ...]) and ask them in a batch:
```bash
drctl rag ask-many --questions-path="<path to file with questions>" --output-path="<path to file to save answers>" --es-index="<index-name>"
```

Format of an input file with questions:
```json
[
    {
        "question": "..."
    }, 
    ...
]
```

## Docker

To build a Docker image, run:
```bash
./bin/docker-cli.sh build
```

To push the image to the registry, run:
```bash
./bin/docker-cli.sh push
```

## Elasticsearch Deployment

To deploy an Elasticsearch on a remote or local node:
1. Copy bin/create-es-volume.sh to the target node.
2. Edit this file and set correct path for a volume to store Elasticsearch data.
3. Run the script. It will create a volume that will be used by Elasticsearch.
4. Copy config/compose.yaml to the target node.
5. Run `docker compose up -d` to start Elasticsearch.

Note: on d.dgx the compose file of ES is currently located in /usr/local/lib/llm-serving/compose-kg-es

