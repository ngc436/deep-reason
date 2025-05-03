from langchain_core.language_models import BaseChatModel

class KGConstructionAgentV2:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def build_wf(self):
        pass
