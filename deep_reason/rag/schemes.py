from typing import List
from pydantic import BaseModel, Field


class ExtractedKeywords(BaseModel):
    keywords: List[str] = Field(..., description="Список выделенных из вопроса поисковых термов для поиска в базе данных ElasticSearch")

    def unique_keywords(self) -> List[str]:
        return list(set(self.keywords))


class PlannedQuestions(BaseModel):
    questions: List[str] = Field(..., description="Список сформированных запросов для поиска в базе данных ElasticSearch")


class ReRankedDocument(BaseModel):
    explanation: str = Field(..., description="Объяснение, почему данный фрагмент контекста является релевантным или НЕ релевантным для ответа на данный вопрос")
    score: float = Field(..., description="Оценка (в диапазоне от 0 до 10 включительно) того, "
                                          "насколько данные фрагмент контекста релевантен для ответа на данный вопрос")
