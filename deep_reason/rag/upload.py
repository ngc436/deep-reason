import json
import logging
import os
import uuid
from typing import List, Optional, Tuple
import numpy as np
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import yaml
from elasticsearch import helpers, AsyncElasticsearch
from tqdm.asyncio import tqdm as atqdm


logger = logging.getLogger(__name__)


async def create_index_if_not_exists(es_client: AsyncElasticsearch, index_name: str, index_body_path: str = "en-es-index-body.yaml"):
    es_index_body_path = os.path.join(os.path.dirname(__file__), index_body_path)
    with open(es_index_body_path, 'r') as f:
        index_body = yaml.safe_load(f)

    if not await es_client.indices.exists(index=index_name):
        await es_client.indices.create(index=index_name, body=index_body)


async def aupload_docs(
        es_client: AsyncElasticsearch,
        index_name: str,
        chunks: List[Document],
        embeddings: List[np.ndarray],
        dataset_name: str,
        loading_batch_size: int = 1000,
        index_body_path: str = "en-es-index-body.yaml"
):
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks and embeddings must have the same length, but got len(chunks)=", len(chunks), 
            "and len(embeddings)=", len(embeddings)
        )

    await create_index_if_not_exists(es_client=es_client, index_name=index_name, index_body_path=index_body_path)

    prepared_chunks = [
        {
            "vector": embedding,
            "metadata": {
                "idx": chunk.metadata["chunk_id"],
                "chapter": "",
                "file_name": chunk.metadata["document_id"],
                "source": dataset_name,
            },
            "paragraph": chunk.page_content
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    errors, success = [], []
    progress_bar = atqdm(total=len(prepared_chunks), desc="Uploading documents")
    async for ok, result in helpers.async_streaming_bulk(
            es_client,
            prepared_chunks,
            index=index_name,
            chunk_size=loading_batch_size,
            raise_on_error=False,
            raise_on_exception=False,
    ):
        progress_bar.update(1)
        if not ok:
            action, error = result.popitem()
            logger.error(f'Failed to index document: {action}, with error: {error}')
            errors.append({'action': action, 'error': error})
        else:
            action, created_doc = result.popitem()
            success.append(created_doc['_id'])
    
    progress_bar.close()
    return errors, success


def load_dataset_as_docs(dataset: str, dataset_path: Optional[str] = None) -> List[Document]:
    known_datasets = {
        "ObliQA": os.path.join("resources", "data", "ObliQA", "ObliQA_train.json"),
    }
    dataset_path = dataset_path or known_datasets.get(dataset, None)

    match dataset:
        case "ObliQA":
            if os.path.isdir(dataset_path):
                json_files = [
                    os.path.join(dataset_path, file) 
                    for file in os.listdir(dataset_path) 
                    if os.path.isfile(os.path.join(dataset_path, file)) and file.endswith(".json")
                ]
            else:
                json_files = [dataset_path]

            # Track seen passage IDs to avoid duplicates
            seen_uids = set()
            documents = []

            for json_file in json_files:
                with open(json_file, 'r') as f:
                    passages = json.load(f)
                
                for passage in passages:
                    uid = passage['ID']
                        
                    # Skip if we've already seen this passage ID
                    if uid in seen_uids:
                        continue
                        
                    seen_uids.add(uid)
                    documents.append(
                        Document(
                            page_content=passage['Passage'] or "   ",
                            metadata={
                                "document_id": passage['DocumentID'],
                                "chunk_id": passage['PassageID']
                            }
                        )
                    )
            
            logger.info(f"Loaded {len(documents)} unique documents from {dataset} dataset")
        case _:
            raise ValueError(f"Dataset {dataset} is not known. Only {known_datasets.keys()} are supported.")

    return documents


async def load_and_upload_dataset(*, 
                                  dataset: str, 
                                  dataset_path: Optional[str] = None,
                                  es_index: str,
                                  es_host: str, 
                                  es_basic_auth: Tuple[str, str], 
                                  embedding_model: str, 
                                  embedding_base_url: str, 
                                  embedding_api_key: str, 
                                  index_body_path: str):
    logger.info(f"Starting to load and upload dataset: {dataset}")
    openai_embedder = OpenAIEmbeddings(
        model=embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        show_progress_bar=True
    )

    async with AsyncElasticsearch(hosts=es_host, basic_auth=es_basic_auth) as es_client:
        op_uuid = uuid.uuid4()
        logger.info(f"Generated operation UUID: {op_uuid}")

        logger.info(f"Loading documents from dataset: {dataset}")
        transformed_docs = load_dataset_as_docs(dataset, dataset_path)

        logger.info(f"Embedding chunks from dataset: {dataset}. Number of chunks: {len(transformed_docs)}")

        emb_docs = await openai_embedder.aembed_documents([doc.page_content for doc in transformed_docs], chunk_size=100)
        embeddings = [np.array(emb) for emb in emb_docs]

        logger.info(f"Uploading documents to Elasticsearch for dataset: {dataset}")
        
        errors, success = await aupload_docs(
            es_client=es_client, 
            index_name=es_index, 
            chunks=transformed_docs, 
            embeddings=embeddings, 
            dataset_name=dataset,
            index_body_path=index_body_path
        )
        
        logger.info(f"Upload complete. Successful: {len(success)}, Errors: {len(errors)}")


async def do_drop_index(es_index: str,
                        es_host: str,
                        es_basic_auth: Tuple[str, str]):
    async with AsyncElasticsearch(hosts=es_host, basic_auth=es_basic_auth) as es_client:
        logger.info(f"Dropping index: {es_index}")
        await es_client.indices.delete(index=es_index, ignore=[400, 404])
        logger.info(f"Index dropped: {es_index}")

