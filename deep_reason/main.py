import asyncio
import logging
from typing import Optional
import click
import pandas as pd

from deep_reason.rag.pipeline import run_rag_pipeline
from deep_reason.rag.upload import do_drop_index, load_and_upload_dataset
from deep_reason.utils import parse_basic_auth


@click.group()  
def cli():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
######################################################
@cli.group()
def rag():
    pass

@rag.command()
@click.option("--question", type=str, required=True, help="Question to ask")
@click.option("--tokenizer-path", type=str, default="resources/qwen2-72b-model-tokenizer", help="Tokenizer path")
@click.option("--es-index", type=str, default="test", help="Elasticsearch index")
@click.option("--es-host", type=str, default="http://d.dgx:9205", help="Elasticsearch host")
@click.option("--es-basic-auth", type=str, default="elastic:admin", help="Elasticsearch API key")
@click.option("--openai-model", type=str, default="/model", help="Embedding model")
@click.option("--openai-base-url", type=str, default="http://d.dgx:8012/v1", help="Embedding base URL")
@click.option("--openai-api-key", type=str, default="token-abc123", help="Embedding API key")
@click.option("--embedding-model", type=str, default="/model", help="Embedding model")
@click.option("--embedding-base-url", type=str, default="http://d.dgx:8029/v1", help="Embedding base URL")
@click.option("--embedding-api-key", type=str, default="token-abc123", help="Embedding API key")
def ask(question: str, 
        tokenizer_path: str,
        es_index: str, es_host: str, es_basic_auth: str, 
        openai_model: str, openai_base_url: str, openai_api_key: str, 
        embedding_model: str, embedding_base_url: str, embedding_api_key: str):
    es_basic_auth = parse_basic_auth(es_basic_auth)
    
    final_states = asyncio.run(
        run_rag_pipeline(
            questions=[question],
            tokenizer_path=tokenizer_path,
            es_index=es_index,
            es_host=es_host,
            es_basic_auth=es_basic_auth,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
        )
    )
    print(f"Answer:\n{final_states[0].answer}")


@rag.command()
@click.option("--questions-path", type=str, required=True, help="Path to a file with questions in JSON format [{'question': '...'}, ...]")
@click.option("--tokenizer-path", type=str, default="resources/qwen2-72b-model-tokenizer", help="Tokenizer path")
@click.option("--es-index", type=str, default="test", help="Elasticsearch index")
@click.option("--es-host", type=str, default="http://d.dgx:9205", help="Elasticsearch host")
@click.option("--es-basic-auth", type=str, default="elastic:admin", help="Elasticsearch API key")
@click.option("--openai-model", type=str, default="/model", help="Embedding model")
@click.option("--openai-base-url", type=str, default="http://d.dgx:8012/v1", help="Embedding base URL")
@click.option("--openai-api-key", type=str, default="token-abc123", help="Embedding API key")
@click.option("--embedding-model", type=str, default="/model", help="Embedding model")
@click.option("--embedding-base-url", type=str, default="http://d.dgx:8029/v1", help="Embedding base URL")
@click.option("--embedding-api-key", type=str, default="token-abc123", help="Embedding API key")
def ask_many(questions_path: str, 
             tokenizer_path: str,
             es_index: str, es_host: str, es_basic_auth: str, 
             openai_model: str, openai_base_url: str, openai_api_key: str, 
             embedding_model: str, embedding_base_url: str, embedding_api_key: str):
    questions = pd.read_json(questions_path)["question"].tolist()
    
    es_basic_auth = parse_basic_auth(es_basic_auth)

    final_states = asyncio.run(
        run_rag_pipeline(
            questions=questions,
            tokenizer_path=tokenizer_path,
            es_index=es_index,
            es_host=es_host,
            es_basic_auth=es_basic_auth,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
        )
    )
    print(f"Answer:\n{final_states[0].answer}")


######################################################
@cli.group()
def es():
    pass


@es.command()
@click.option("--dataset", type=str, help="Dataset to load and upload")
@click.option("--dataset-path", type=str, default=None, help="Path to dataset")
@click.option("--es-index", type=str, default="test", help="Elasticsearch index")
@click.option("--es-host", type=str, default="http://d.dgx:9205", help="Elasticsearch host")
@click.option("--es-basic-auth", type=str, default="elastic:admin", help="Elasticsearch API key")
@click.option("--embedding-model", type=str, default="/model", help="Embedding model")
@click.option("--embedding-base-url", type=str, default="http://d.dgx:8029/v1", help="Embedding base URL")
@click.option("--embedding-api-key", type=str, default="token-abc123", help="Embedding API key")
@click.option("--index-body-path", type=str, default="en-es-index-body.yaml", help="Index body path")
def upload(dataset: str, dataset_path: Optional[str], es_index: str, es_host: str, es_basic_auth: str, embedding_model: str, embedding_base_url: str, embedding_api_key: str, index_body_path: str):
    es_basic_auth = parse_basic_auth(es_basic_auth)
    
    asyncio.run(
        load_and_upload_dataset(
            dataset=dataset, 
            dataset_path=dataset_path, 
            es_host=es_host, 
            es_basic_auth=es_basic_auth, 
            es_index=es_index,
            embedding_model=embedding_model, 
            embedding_base_url=embedding_base_url, 
            embedding_api_key=embedding_api_key,
            index_body_path=index_body_path
        )
    )


@es.command()
@click.option("--es-index", type=str, default="test", help="Index to drop")
@click.option("--es-host", type=str, default="http://d.dgx:9205", help="Elasticsearch host")
@click.option("--es-basic-auth", type=str, default="elastic:admin", help="Elasticsearch API key")
def drop_index(es_host: str, es_basic_auth: str, es_index: str):
    es_basic_auth = parse_basic_auth(es_basic_auth)
    
    asyncio.run(
        do_drop_index(es_host=es_host, es_basic_auth=es_basic_auth, es_index=es_index)
    )


if __name__ == "__main__":
    cli()

