import logging
import json
import os
from tqdm.asyncio import tqdm
import asyncio
import time

from typing import Annotated, Tuple
from typing import Any
from typing import List
from typing import Optional
from typing import cast


from elasticsearch import AsyncElasticsearch
from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages.ai import AIMessage
from langchain_core.vectorstores import VectorStore
from langchain_elasticsearch.vectorstores import ElasticsearchStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.constants import END
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.pregel.io import AddableValuesDict
from pydantic import BaseModel
from transformers import PreTrainedTokenizer, AutoTokenizer

from deep_reason.rag.prompts import chat_answer_prompt_template
from deep_reason.rag.prompts import keywords_extraction_prompt_template
from deep_reason.rag.prompts import planner_prompt_template
from deep_reason.rag.prompts import reranker_prompt_template
from deep_reason.rag.schemes import ExtractedKeywords
from deep_reason.rag.schemes import PlannedQuestions
from deep_reason.rag.schemes import ReRankedDocument
from deep_reason.rag.utils import MultiQueryRetriever, VLLMChatOpenAI
from deep_reason.rag.utils import _doc2str
from deep_reason.rag.utils import create_structured_chain


logger = logging.getLogger(__name__)


def _take_any(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a or b


class RAGIntermediateOutputs(BaseModel):
    # general
    question: Annotated[str, _take_any]

    # es part
    keywords: Annotated[Optional[List[str]], _take_any] = None
    es_retrieved_documents: Annotated[Optional[List[Document]], _take_any] = None

    # planner part (only for vector store)
    planned_queries: Annotated[Optional[List[str]], _take_any] = None
    retrieved_documents: Annotated[Optional[List[Document]], _take_any] = None

    # reranking part
    context_documents: Annotated[Optional[List[Document]], _take_any] = None
    reranked_documents: Annotated[Optional[List[Document]], _take_any] = None
    reranker_answers: Annotated[Optional[List[str]], _take_any] = None
    reranker_scores: Annotated[Optional[List[float]], _take_any] = None

    # answer generating part
    answer_context: Annotated[Optional[str], _take_any] = None
    answer: Annotated[Optional[str], _take_any] = None

    # general if error occurs anywhere
    error: Annotated[Optional[Exception], _take_any] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def contexts(self) -> Optional[List[Document]]:
        return self.reranked_documents or self.retrieved_documents


class RAGPipelineBuilder:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        tokenizer: PreTrainedTokenizer,
        store: Optional[List[VectorStore] | VectorStore] = None,
        es_client: Optional[AsyncElasticsearch] = None,
        es_collection: Optional[List[str] | str] = None,
        filter: Optional[Any] = None,
        max_input_tokens: int = 24000,
        retrieving_top_k: int = 3,
        document_col_name: str = "paragraph",
        metadata_col_name: str = "metadata",
    ):
        if es_client is None and store is None:
            raise ValueError("es_client and store cannot be None simultaneously")

        if es_client and not es_collection:
            raise ValueError(
                "ElasticSearch collection name should be provided if ElasticSearch client is not None"
            )

        self._llm = llm
        self._tokenizer = tokenizer
        self._stores = [store] if isinstance(store, VectorStore) else store
        self._es_client = es_client
        self._es_collections = (
            [es_collection] if isinstance(es_collection, str) else es_collection
        )
        self._filter = filter
        self._max_input_tokens = max_input_tokens
        self._retrieving_top_k = retrieving_top_k
        self._document_col_name = document_col_name
        self._metadata_col_name = metadata_col_name

    def build_chain(
        self,
        do_vector_search: bool = True,
        do_full_text_search: bool = True,
        do_planning: bool = True,
        do_reranking: bool = True,
    ) -> Runnable[RAGIntermediateOutputs, AddableValuesDict]:
        workflow = StateGraph(RAGIntermediateOutputs)

        if not do_full_text_search and not do_vector_search:
            raise ValueError(
                "At least one of do_full_text_search or do_vector_search must be True"
            )

        if (do_full_text_search and do_vector_search) and not do_reranking:
            raise ValueError(
                "Cannot have both full text search and vector search enabled without reranking"
            )

        # Add nodes based on enabled features
        if do_full_text_search:
            workflow.add_node("keywords_extractor", self._node_aextract_keywords)
            workflow.add_node("retriever_es", self._node_aretrievieng_es_based)

        if do_vector_search:
            if do_planning:
                workflow.add_node("planner", self._node_aplanning)
            workflow.add_node("retriever_vector", self._node_aretrieving_vector_based)

        # Always add these nodes
        workflow.add_node(
            "joiner_retrieved_documents", self._node_ajoin_retrieved_documents
        )

        if do_reranking:
            workflow.add_node("reranker", self._node_areranking)

        workflow.add_node("make_context", self._node_amake_context)
        workflow.add_node("answer_generator", self._node_agenerate_answer)

        # Set up edges based on enabled features
        sources_to_joiner = []

        # Keywords extraction and ES retrieval path
        if do_full_text_search:
            workflow.add_edge(START, "keywords_extractor")
            workflow.add_edge("keywords_extractor", "retriever_es")
            sources_to_joiner.append("retriever_es")

        # Planning and vector retrieval path
        if do_vector_search:
            if do_planning:
                workflow.add_edge(START, "planner")
                workflow.add_edge("planner", "retriever_vector")
            else:
                workflow.add_edge(START, "retriever_vector")
            sources_to_joiner.append("retriever_vector")

        # Connect retrieval sources to joiner
        if sources_to_joiner:
            workflow.add_edge(sources_to_joiner, "joiner_retrieved_documents")
        else:
            # If no retrieval methods are enabled, connect START directly to joiner
            workflow.add_edge(START, "joiner_retrieved_documents")

        # Connect joiner to reranker or directly to make_context
        if do_reranking:
            workflow.add_edge("joiner_retrieved_documents", "reranker")
            workflow.add_edge("reranker", "make_context")
        else:
            workflow.add_edge("joiner_retrieved_documents", "make_context")

        # Final steps remain the same
        workflow.add_edge("make_context", "answer_generator")
        workflow.add_edge("answer_generator", END)

        wf = workflow.compile()
        wf.name = "rag_workflow"
        return wf

    async def _node_aextract_keywords(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> RAGIntermediateOutputs:
        if self._es_client is None:
            return state

        extract_keywords_chain = create_structured_chain(
            prompt=keywords_extraction_prompt_template,
            llm=self._llm,
            parser=PydanticOutputParser(pydantic_object=ExtractedKeywords),
        )

        keywords = cast(
            ExtractedKeywords, 
            await extract_keywords_chain.ainvoke(
                input={"question": state.question}, config=config
            )
        )
        keywords = keywords.unique_keywords() if keywords else []
        logger.info(f"Extracted keywords: {keywords}")
        return state.copy(update={"keywords": keywords})

    async def _node_aretrievieng_es_based(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> RAGIntermediateOutputs:
        documents = []
        for es_collection in self._es_collections:
            if self._filter:
                query = {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": " ".join(state.keywords),
                                "fields": [
                                    self._document_col_name,
                                    f"{self._metadata_col_name}.chapter",
                                    f"{self._metadata_col_name}.file_name",
                                ],
                            }
                        },
                        "filter": self._filter,
                    }
                }
            else:
                query = {
                    "multi_match": {
                        "query": " ".join(state.keywords),
                        "fields": [
                            self._document_col_name,
                            f"{self._metadata_col_name}.chapter",
                            f"{self._metadata_col_name}.file_name",
                        ],
                    }
                }

            obj = await self._es_client.search(
                index=es_collection, size=self._retrieving_top_k, query=query
            )
            # TODO: handle if no results
            documents.extend(
                [
                    Document(
                        page_content=r["_source"]["paragraph"],
                        metadata=r["_source"]["metadata"],
                    )
                    for r in obj.body["hits"].get("hits", [])
                ]
            )

        logger.info(f"Retrieved documents (ES search): {len(documents)}")

        return state.copy(update={"es_retrieved_documents": documents})

    async def _node_aplanning(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> RAGIntermediateOutputs:
        planner_chain = create_structured_chain(
            prompt=planner_prompt_template,
            llm=self._llm,
            parser=PydanticOutputParser(pydantic_object=PlannedQuestions),
        )

        queries = cast(
            PlannedQuestions, 
            await planner_chain.ainvoke(
                {"question": state.question}, config=config
            )
        )
        queries = queries.questions if queries else []
        queries = list({state.question, *queries})

        logger.info(f"Planned queries: {queries}")

        return state.copy(update={"planned_queries": queries})

    async def _node_aretrieving_vector_based(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> RAGIntermediateOutputs:
        if self._stores is None:
            return state

        def _get_search_kwargs(store: VectorStore):
            search_kwargs = {"k": self._retrieving_top_k}

            if self._filter and isinstance(store, ElasticsearchStore):
                search_kwargs["filter"] = self._filter

            return search_kwargs

        mq_retriever = MultiQueryRetriever(
            retriever=[
                store.as_retriever(search_kwargs=_get_search_kwargs(store))
                for store in self._stores
            ]
        )

        retriever_chain = RunnableLambda(
            func=mq_retriever.do_batch, afunc=mq_retriever.do_abatch
        )

        queries = state.planned_queries or [state.question]

        logger.info(f"Performing vector search with queries: {queries}")

        docs = await retriever_chain.ainvoke(queries, config=config)
        logger.info(f"Retrieved documents (Vector search): {len(docs)}")
        return state.copy(update={"retrieved_documents": docs})

    async def _node_ajoin_retrieved_documents(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> RAGIntermediateOutputs:
        docs = [
            *(state.es_retrieved_documents or []),
            *(state.retrieved_documents or []),
        ]

        if not docs:
            raise ValueError("No documents retrieved")

        return state.copy(update={"context_documents": docs})

    async def _node_areranking(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> AddableValuesDict:
        reranker_chain = create_structured_chain(
            prompt=reranker_prompt_template,
            llm=self._llm,
            parser=PydanticOutputParser(pydantic_object=ReRankedDocument),
        )

        reranked_scores = await reranker_chain.abatch(
            inputs=[
                {"question": state.question, "context": doc.page_content}
                for doc in state.context_documents
            ],
            return_exceptions=True,
            config=config
        )

        # Log errors if they exist and raise an exception if all results are errors
        exceptions = [
            score for score in reranked_scores if isinstance(score, Exception)
        ]
        if exceptions:
            for i, exc in enumerate(exceptions):
                logger.error(f"Error during reranking (document {i}): {exc}")

        valid_results = [
            (doc, rscore)
            for doc, rscore in zip(state.context_documents, reranked_scores)
            if isinstance(rscore, ReRankedDocument)
        ]

        if not valid_results:
            error_msg = (
                f"All reranking operations failed. Total errors: {len(exceptions)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        reranked_docs = sorted(valid_results, key=lambda x: x[1].score, reverse=True)

        reranked_scores = [
            (doc, rscore) for doc, rscore in reranked_docs if rscore.score > 2
        ]

        docs = [doc for doc, _ in reranked_docs]
        scores = [rscore.score for _, rscore in reranked_docs]
        reranker_answers = [rscore.explanation for _, rscore in reranked_docs]

        return state.copy(
            update={
                "reranked_documents": docs,
                "reranker_answers": reranker_answers,
                "reranker_scores": scores,
            }
        )

    async def _node_amake_context(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> RAGIntermediateOutputs:
        documents = state.reranked_documents or state.context_documents
        paragraphs = [
            f"Источник {i + 1}: {_doc2str(doc)}" for i, doc in enumerate(documents)
        ]

        curr_ctx = []
        for paragraph in paragraphs:
            curr_ctx.append(paragraph)
            ctx = "\n".join(curr_ctx)
            prompt_value = await chat_answer_prompt_template.ainvoke(
                input={
                    "question": state.question,
                    "context": ctx,
                }
            )
            prompt_value = cast(ChatPromptValue, prompt_value).to_string()
            prompt_tokens_len = len(self._tokenizer.tokenize(prompt_value))
            if prompt_tokens_len > self._max_input_tokens:
                break

        if len(curr_ctx) == 0:
            raise ValueError("Context is too long")
        elif len(curr_ctx) < len(paragraphs):
            logger.warning(
                "Reducing number of paragraphs in the context due to limit on tokens."
                f"Took %s paragraphs out of %s." % (len(curr_ctx), len(paragraphs))
            )

        paragraphs_contexts = "\n".join(curr_ctx)

        return state.copy(update={"answer_context": paragraphs_contexts})

    async def _node_agenerate_answer(
        self, state: RAGIntermediateOutputs, config: RunnableConfig
    ) -> RAGIntermediateOutputs:
        answer_generator_chain = chat_answer_prompt_template | self._llm

        answer = cast(
            AIMessage,
            await answer_generator_chain.ainvoke(
                input={
                    "question": state.question,
                    "context": state.answer_context,
                },
                config=config,
            )
        )

        final_answer = answer.content

        logger.info(f"FINAL ANSWER: {final_answer}")

        return state.copy(update={"answer": final_answer})
    

async def run_rag_pipeline(*, 
                           questions: List[str], 
                           tokenizer_path: str,
                           es_index: str, es_host: str, es_basic_auth: Tuple[str, str], 
                           openai_model: str, openai_base_url: str, openai_api_key: str,
                           embedding_model: str, embedding_base_url: str, embedding_api_key: str,
                           do_vector_search: bool = True, do_full_text_search: bool = True,
                           do_planning: bool = True, do_reranking: bool = True,
                           max_concurrency: int = 20,
                           cache_file: Optional[str] = None,
                           batch_size: int = 100,
                           task_timeout: int = 300,
                           max_retries: int = 3) -> List[RAGIntermediateOutputs]:
    logger.info(f"Starting RAG pipeline with {len(questions)} questions")
    logger.info(f"Configuration: vector_search={do_vector_search}, full_text_search={do_full_text_search}, planning={do_planning}, reranking={do_reranking}")
    logger.info(f"Max concurrency: {max_concurrency}, Batch size: {batch_size}, Task timeout: {task_timeout}s, Max retries: {max_retries}")
    logger.info(f"Cache file: {cache_file}")
    
    # Initialize cache dictionary
    cache = {}
    
    # Load existing cache if file exists
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cache from {cache_file}")
        with open(cache_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        # Parse the JSON data
                        data = json.loads(line.strip())
                        # Create a new RAGIntermediateOutputs object with the parsed data
                        # We need to handle the case where some fields might be missing
                        cached_result = RAGIntermediateOutputs(
                            question=data.get("question", ""),
                            keywords=data.get("keywords", None),
                            es_retrieved_documents=None,  # We don't cache these complex objects
                            planned_queries=data.get("planned_queries", None),
                            retrieved_documents=None,  # We don't cache these complex objects
                            context_documents=None,  # We don't cache these complex objects
                            reranked_documents=None,  # We don't cache these complex objects
                            reranker_answers=data.get("reranker_answers", None),
                            reranker_scores=data.get("reranker_scores", None),
                            answer_context=data.get("answer_context", None),
                            answer=data.get("answer", None),
                            error=None
                        )
                        cache[cached_result.question] = cached_result
                    except Exception as e:
                        logger.warning(f"Failed to load cached result: {e}")
    
    # Determine which questions need processing
    questions_to_process = []
    for question in questions:
        if question not in cache:
            questions_to_process.append(question)
    
    logger.info(f"Found {len(cache)} cached results. Processing {len(questions_to_process)} new questions.")
    
    if not questions_to_process:
        logger.info("No new questions to process. Returning cached results.")
        return [cache[q] for q in questions]
    
    logger.info("Initializing models and connections...")
    es_timeout = 300
    try:
        logger.info(f"Initializing LLM with model={openai_model}, base_url={openai_base_url}")
        llm = VLLMChatOpenAI(
            model=openai_model,
            base_url=openai_base_url,
            api_key=openai_api_key,
            temperature=0.01,
            max_tokens=2048,
        )
        
        # Test LLM availability
        logger.info("Testing LLM availability...")
        try:
            test_response = await llm.ainvoke("Test message")
            logger.info("LLM is available and responding")
        except Exception as e:
            logger.error(f"LLM is not available: {str(e)}")
            raise

        logger.info(f"Initializing embedding model with model={embedding_model}, base_url={embedding_base_url}")
        embedding_model = OpenAIEmbeddings(
            model=embedding_model,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
        )
        
        # Test embedding model availability
        logger.info("Testing embedding model availability...")
        try:
            test_embedding = await embedding_model.aembed_query("Test message")
            logger.info("Embedding model is available and responding")
        except Exception as e:
            logger.error(f"Embedding model is not available: {str(e)}")
            raise

        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded successfully")

        logger.info(f"Connecting to Elasticsearch at {es_host}")
        async with AsyncElasticsearch(
            hosts=es_host,
            basic_auth=es_basic_auth,
            timeout=es_timeout
        ) as es_client:
            # Test Elasticsearch connection
            logger.info("Testing Elasticsearch connection...")
            try:
                health = await es_client.cluster.health()
                logger.info(f"Elasticsearch is available. Cluster health: {health}")
            except Exception as e:
                logger.error(f"Elasticsearch is not available: {str(e)}")
                raise
            
            logger.info(f"Initializing ElasticsearchStore with index={es_index}")
            es_store = ElasticsearchStore(
                embedding=embedding_model,
                index_name=es_index,
                es_url=es_host,
                es_user=es_basic_auth[0],
                es_password=es_basic_auth[1],
                query_field="paragraph",
                es_params={"timeout": es_timeout}
            )
            logger.info("ElasticsearchStore initialized successfully")

            logger.info("Building RAG pipeline chain")
            builder = RAGPipelineBuilder(
                llm=llm,
                tokenizer=tokenizer,
                store=es_store,
                es_client=es_client,
                es_collection=es_index,
            )           
            chain = builder.build_chain(do_planning=do_planning, do_reranking=do_reranking, 
                                        do_vector_search=do_vector_search, do_full_text_search=do_full_text_search)
            logger.info("RAG pipeline chain built successfully")
            logger.info(f"Chain configuration: planning={do_planning}, reranking={do_reranking}, vector_search={do_vector_search}, full_text_search={do_full_text_search}")
            
            # Setup progress bar
            pbar = tqdm(total=len(questions_to_process), desc="Processing questions") 
            
            # File to append results as they come in
            if cache_file:
                cache_dir = os.path.dirname(os.path.abspath(cache_file))
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"Will save results to cache file: {cache_file}")
            
            # Process questions in batches
            processed_count = 0
            failed_questions = []
            
            # Function to process a batch of questions with timeout and retry logic
            async def process_batch(batch_questions):
                nonlocal processed_count
                tasks = []
                
                for question in batch_questions:
                    task = asyncio.create_task(
                        asyncio.wait_for(
                            chain.ainvoke(
                                RAGIntermediateOutputs(question=question),
                                config=RunnableConfig(max_concurrency=max_concurrency)
                            ),
                            timeout=task_timeout
                        )
                    )
                    tasks.append((question, task))
                
                results = []
                for question, task in tasks:
                    retries = 0
                    while retries < max_retries:
                        try:
                            final_state = await task
                            result = RAGIntermediateOutputs.model_validate(final_state)
                            cache[result.question] = result
                            processed_count += 1
                            
                            # Write to cache file
                            if cache_file:
                                with open(cache_file, 'a') as f:
                                    # Convert to a serializable format, handling non-serializable types
                                    serializable_data = result.model_dump(exclude={"error"})
                                    # Ensure all values are JSON serializable
                                    for key, value in serializable_data.items():
                                        if isinstance(value, (Document, AIMessage)):
                                            serializable_data[key] = str(value)
                                        elif isinstance(value, list):
                                            serializable_data[key] = [
                                                str(item) if isinstance(item, (Document, AIMessage)) else item
                                                for item in value
                                            ]
                                    f.write(json.dumps(serializable_data) + '\n')
                            
                            # Update progress bar
                            pbar.update(1)
                            
                            results.append(result)
                            break
                        except asyncio.TimeoutError:
                            retries += 1
                            logger.warning(f"Task for question '{question[:50]}...' timed out after {task_timeout}s (attempt {retries}/{max_retries})")
                            if retries >= max_retries:
                                logger.error(f"Failed to process question after {max_retries} attempts: '{question[:50]}...'")
                                failed_questions.append(question)
                                pbar.update(1)
                        except Exception as e:
                            retries += 1
                            logger.error(f"Error processing question '{question[:50]}...' (attempt {retries}/{max_retries}): {str(e)}")
                            if retries >= max_retries:
                                logger.error(f"Failed to process question after {max_retries} attempts: '{question[:50]}...'")
                                failed_questions.append(question)
                                pbar.update(1)
                
                return results
            
            # Process questions in batches
            all_results = []
            for i in range(0, len(questions_to_process), batch_size):
                batch = questions_to_process[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(questions_to_process) + batch_size - 1)//batch_size} ({len(batch)} questions)")
                batch_start_time = time.time()
                
                batch_results = await process_batch(batch)
                all_results.extend(batch_results)
                
                batch_time = time.time() - batch_start_time
                logger.info(f"Completed batch {i//batch_size + 1} in {batch_time:.2f}s ({len(batch_results)}/{len(batch)} successful)")
                
                # Add a small delay between batches to allow resources to be freed
                if i + batch_size < len(questions_to_process):
                    await asyncio.sleep(2)
            
            pbar.close()
            logger.info(f"Completed processing {processed_count} questions")
            if failed_questions:
                logger.warning(f"Failed to process {len(failed_questions)} questions")
                logger.warning(f"Failed questions: {failed_questions[:10]}...")
    
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
        raise
    
    # Reorder results to match input questions
    logger.info("Reordering results to match input questions")
    final_states = []
    for question in questions:
        if question not in cache:
            logger.error(f"Question not found in results: {question}")
            # Create a placeholder result for failed questions
            placeholder = RAGIntermediateOutputs(
                question=question,
                error=None  # We can't serialize Exception objects, so set to None
            )
            # Add a note in the answer field to indicate failure
            placeholder.answer = f"Failed to process this question due to resource limitations or timeouts."
            final_states.append(placeholder)
        else:
            final_states.append(cache[question])
    
    logger.info(f"RAG pipeline completed. Returning {len(final_states)} results")
    return final_states

