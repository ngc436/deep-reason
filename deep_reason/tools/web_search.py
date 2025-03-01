from deep_reason.utils import StreamQAPipeline

class WebSearchPipeline(StreamQAPipeline[WebIntermediateOutputs]):
    def __init__(self,
                 agent_id: str,
                 model: str,
                 model_type: ModelType = ModelType.llama3,
                 fast_search: bool = True,
                 report_type: str = "research_report",
                 retrievers: str = "duckduckgo",
                 embeddings: Embeddings = None,
                 tokenizer: str = "/home/cunning/llama-3/Meta-Llama-3-8B",
                 max_input_tokens: int = 6144,
                 max_chat_history_token_length: int = 24576,
                 model_kwargs: Optional[Dict[str, Any]] = None,
                 acronyms_filepath: Optional[str] = None,
                 langfuse_handler: Optional[CallbackHandler] = None,
                 runnable_config: Optional[RunnableConfig] = None,
                 ) -> None:
        self._agent_id = agent_id
        self._model = model
        self._model_type = model_type
        self._model_kwargs = model_kwargs 
        # or {
        #     "temperature": 0,
        #     "top_p": 0.95,
        #     "max_tokens": 1024,
        #     "openai_api_key": "token-abc123",
        #     "openai_api_base": f"http://a.dgx:50080/qwen2-72b/v1"
        # }
        self._report_type = report_type
        self._gpt_researcher = None
        self._max_input_tokens = max_input_tokens
        self._max_chat_history_token_length = max_chat_history_token_length
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        max_iterations = 3
        max_search_results_per_query = 5
        if fast_search:
            max_iterations = 2
            max_search_results_per_query = 3

        cfg = CustomConfig(
            openai_base_url=self._model_kwargs['openai_api_base'],
            openai_api_key=self._model_kwargs['openai_api_key'],
            tokenizer_path=tokenizer,
            retrievers=retrievers,
            model_kwargs=model_kwargs,
            max_iterations=max_iterations,
            max_search_results_per_query=max_search_results_per_query,
            embeddings=embeddings
        )

        self._researcher_config = cfg
        self._acronyms_filepath = acronyms_filepath
        self._langfuse_handler = langfuse_handler
        self._runnable_config = runnable_config

        self._final_state: Optional[WebIntermediateOutputs] = None

    @staticmethod
    def _convert_pipeline_result(agent_id: str, question: str, result: AddableValuesDict) -> WebIntermediateOutputs:
        return WebIntermediateOutputs(agent_id=agent_id, question=question, error=result) \
            if isinstance(result, Exception) else WebIntermediateOutputs(** result)

    def _get_planner_llm(self) -> BaseLLM:
        llm = NonStreamableVLLMOpenAI(
            name="planner",
            model=self._model,
            streaming=False,
            **self._model_kwargs
        )
        llm = llm.bind(stop=["<|eot_id|>"])
        return cast(BaseLLM, llm)

    def _check_num_input_tokens(self, question: str):
        tokens = self._tokenizer.tokenize(question)
        if len(tokens) > self._max_input_tokens:
            raise WebPipelineMaxInputTokensExceededException(
                f"Input length {len(tokens)} exceeds allowed max number of tokens {self._max_input_tokens}."
            )

    def _get_llm_prompt_converter(self) -> Runnable[PromptValue, StringPromptValue]:
        if self._model_type == ModelType.llama3:
            converter = RunnableLambda(llama3_convert, afunc=allama3_convert)
        else:
            converter = RunnableLambda(qwen2_convert, afunc=aqwen2_convert)

        return converter

    def _get_contextualizer_chain(self):
        chain = (
            contextualize_q_prompt
            | self._get_llm_prompt_converter()
            | self._get_planner_llm()
        )
        return chain

    # for compatibility with RAGPipeline and other pipelines

    @property
    def final_state(self):
        return self._final_state 

    async def batch(self,
                    questions: List[str],
                    raise_if_error: bool = False,
                    show_progress: bool = True,
                    max_concurrency: Optional[int] = None) -> List[WebIntermediateOutputs]:
        chain = self.build_chain()

        for question in questions:
            self._check_num_input_tokens(question)

        with tqdm(total=len(questions), disable=not show_progress, desc="Web") as pbar:
            ioutputs = WebIntermediateOutputsCallback(
                total=len(questions),
                trigger_nodes=[WEB_COMPONENT_ANSWER_GENERATOR],
                pbar=pbar
            )
            # we will retrieve all results through callback

            callbacks = [
                ioutputs, self._langfuse_handler] if self._langfuse_handler else [ioutputs]
            results = await chain.abatch(
                [WebIntermediateOutputs(agent_id=self._agent_id, question=question)
                 for question in questions],
                config=RunnableConfig(
                    callbacks=callbacks, max_concurrency=max_concurrency),
                return_exceptions=not raise_if_error
            )
        return [
            self._convert_pipeline_result(self._agent_id, question, result)
            for question, result in zip(questions, results)
        ]

    async def stream(self,
                     question: str,
                     chat_history: Optional[List[BaseMessage]
                                            | List[Tuple[str, str]]] = None,
                     raise_if_error: bool = False,
                     only_eos_answer_event: bool = False) -> AsyncIterator[PipelineEvent]:
        # planning, retrieving, reranking, answering
        current_output = ''
        chain = self.build_chain()
        try:
            self._check_num_input_tokens(question)
            chat_history = self._convert_chat_history(chat_history)
            chat_history = self._cut_chat_history(
                chat_history, self._max_chat_history_token_length, self._tokenizer)

            conf = self._runnable_config if self._runnable_config else {
                "callbacks": [self._langfuse_handler]} if self._langfuse_handler else None
            async for event in chain.astream_events(
                    WebIntermediateOutputs(agent_id=self._agent_id,
                                           question=question,
                                           chat_history=chat_history),
                    config=conf,
                    version="v2"
            ):
                all_events = [
                    WEB_COMPONENT_QUESTION_CONTEXTUALIZER,
                    WEB_COMPONENT_ACRONYMS_EXTRACTOR,
                    WEB_COMPONENT_MAKE_CONTEXT,
                    WEB_COMPONENT_ANSWER_GENERATOR,
                ]
                if event['event'] == 'on_chain_end':
                    state = cast(WebIntermediateOutputs,
                                 event['data']['output'])
                    if event['name'] in all_events and state.error:
                        stream_event = WebErrorEvent(agent_id=self._agent_id,
                                                     error=str(state.error),
                                                     is_eos=True)
                    elif event['name'] == WEB_COMPONENT_QUESTION_CONTEXTUALIZER:
                        stream_event = WebContextualizerEvent(agent_id=self._agent_id,
                                                              contextualize_questions=state.contextualized_question)
                    elif event['name'] == WEB_COMPONENT_MAKE_CONTEXT:
                        stream_event = WebSearchingEvent(agent_id=self._agent_id,
                                                         context=state.context_documents)
                    elif event['name'] == WEB_COMPONENT_ANSWER_GENERATOR:
                        stream_event = AnsweringEvent(agent_id=self._agent_id,
                                                      name="web_answer",
                                                      output=state.answer,
                                                      delta=state.answer,
                                                      is_eos=True
                                                      )
                    elif event['name'] == WEB_COMPONENT_WORKFLOW:
                        logger.debug(f"last event: {event}")
                        self._final_state = WebIntermediateOutputs.parse_obj(
                            event['data']['output'])
                        stream_event = None
                    else:
                        stream_event = None
                elif event['event'] == 'on_llm_stream' and event['name'] == WEB_COMPONENT_ANSWER_GENERATOR \
                        and not only_eos_answer_event:
                    delta = event['data']['chunk'].text
                    current_output += delta
                    stream_event = AnsweringEvent(agent_id=self._agent_id,
                                                  output=current_output,
                                                  delta=delta,
                                                  is_eos=False)
                else:
                    stream_event = None

                if stream_event:
                    if not isinstance(stream_event, AnsweringEvent) or stream_event.is_eos:
                        logger.debug(f"Yielding stream event: {stream_event}")

                    yield stream_event
        except Exception as ex:
            yield WebErrorEvent(agent_id=self._agent_id, error=str(ex), is_eos=True)
            if raise_if_error:
                raise ex

    async def invoke(self,
                     question: str,
                     chat_history: Optional[List[BaseMessage]
                                            | List[Tuple[str, str]]] = None,
                     raise_if_error: bool = False) -> WebIntermediateOutputs:
        chain = self.build_chain()

        self._check_num_input_tokens(question)
        chat_history = self._convert_chat_history(chat_history)
        chat_history = self._cut_chat_history(
            chat_history, self._max_chat_history_token_length, self._tokenizer)

        conf = self._runnable_config if self._runnable_config else {
            "callbacks": [self._langfuse_handler]} if self._langfuse_handler else None
        # we will retrieve all results through callback
        result = await chain.ainvoke(
            WebIntermediateOutputs(agent_id=self._agent_id,
                                   question=question,
                                   chat_history=chat_history),
            config=conf,
            # return_exceptions=not raise_if_error
        )
        self._final_state = self._convert_pipeline_result(self._agent_id, question, result)
        return self.final_state

    async def events(self,
                     question: str,
                     chat_history: Optional[List[BaseMessage]
                                            | List[Tuple[str, str]]] = None,
                     raise_if_error: bool = False,
                     only_eos_answer_event: bool = False) -> List[PipelineEvent]:
        return [event async for event in self.stream(question, chat_history, raise_if_error, only_eos_answer_event)]

    def build_chain(self) -> Runnable[WebIntermediateOutputs, AddableValuesDict]:

        def call_with(
            func: Callable[[WebIntermediateOutputs, RunnableConfig],
                           Awaitable[WebIntermediateOutputs]]
        ) -> Callable[[WebIntermediateOutputs], Awaitable[WebIntermediateOutputs]]:
            async def _func(state: WebIntermediateOutputs, config: RunnableConfig):
                return await func(state, config)

            return cast(Callable[[WebIntermediateOutputs], Awaitable[WebIntermediateOutputs]], _func)

        contextualizer_chain = self._get_contextualizer_chain()
        acronyms_chain = AcronymsExtractor(self._acronyms_filepath)

        async def _acontextualize_question(state: WebIntermediateOutputs) -> WebIntermediateOutputs:
            if state.chat_history:
                contextualized_question = await contextualizer_chain.ainvoke(
                    input={"question": state.question,
                           "chat_history": state.chat_history},
                )
            else:
                contextualized_question = state.question
            return state.copy(update={'contextualized_question': contextualized_question})

        async def _aextract_acronyms(state: WebIntermediateOutputs) -> WebIntermediateOutputs:
            acronyms_expansion = acronyms_chain(
                state.contextualized_question or state.question)
            return state.copy(update={'acronyms_expansion': acronyms_expansion})

        async def _aresearch(state: WebIntermediateOutputs) -> WebIntermediateOutputs:
            query = (state.contextualized_question or state.question)
            if state.acronyms_expansion:
                query = query + '\nВозможные акронимы: ' + state.acronyms_expansion
            self._gpt_researcher = GPTResearcher(
                query, self._researcher_config, self._report_type)
            research_result = await self._gpt_researcher.conduct_research()
            return state.copy(update={'context_documents': research_result})

        async def _awrite_report(state: WebIntermediateOutputs, config: RunnableConfig) -> WebIntermediateOutputs:
            llm = VLLMOpenAI(
                name=WEB_COMPONENT_ANSWER_GENERATOR,
                model=self._model,
                streaming=True,
                **self._model_kwargs
            )
            llm = llm.bind(stop=["<|eot_id|>"])
            report = await self._gpt_researcher.write_report(stream_llm=llm, runnable_config=config)
            return state.copy(update={'answer': report})

        workflow = StateGraph(WebIntermediateOutputs)

        # assemble the chain

        workflow.add_node(WEB_COMPONENT_QUESTION_CONTEXTUALIZER,
                          _acontextualize_question)
        workflow.add_node(WEB_COMPONENT_ACRONYMS_EXTRACTOR, _aextract_acronyms)
        workflow.add_node(WEB_COMPONENT_MAKE_CONTEXT, _aresearch)
        workflow.add_node(WEB_COMPONENT_ANSWER_GENERATOR,
                          call_with(_awrite_report))

        # workflow start, initial question and context preparation
        workflow.add_edge(START, WEB_COMPONENT_QUESTION_CONTEXTUALIZER)
        workflow.add_edge(WEB_COMPONENT_QUESTION_CONTEXTUALIZER,
                          WEB_COMPONENT_ACRONYMS_EXTRACTOR)

        # planner + vector store branch
        workflow.add_edge(WEB_COMPONENT_ACRONYMS_EXTRACTOR,
                          WEB_COMPONENT_MAKE_CONTEXT)

        # answer generating
        workflow.add_edge(WEB_COMPONENT_MAKE_CONTEXT,
                          WEB_COMPONENT_ANSWER_GENERATOR)
        workflow.add_edge(WEB_COMPONENT_ANSWER_GENERATOR, END)

        wf = workflow.compile()
        wf.name = WEB_COMPONENT_WORKFLOW
        return wf