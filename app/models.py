"""
models.py

All the data shapes that get passed around the app live here. Keeping them
in one file makes it easy to see, at a glance, exactly what "structured
output" means for this project - every agent returns one of these instead
of a plain string.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel


class SubTask(BaseModel):
    """One piece of work the orchestrator wants a sub-agent to do."""
    agent: Literal["data", "rag", "comparison"]
    instruction: str


class TaskSpec(BaseModel):
    """
    The full plan the orchestrator builds from a business question.
    This is what gets handed to each sub-agent, instead of just the
    raw question text.
    """
    question: str
    subtasks: List[SubTask]


class ThemeCount(BaseModel):
    theme: str
    count: int
    percentage: float


class DataResult(BaseModel):
    """What DataAgent returns after crunching the survey numbers."""
    response_count: int
    average_rating: float
    csat: float
    top_themes: List[ThemeCount]
    date_range: str


class RagResult(BaseModel):
    """What RAGAgent returns after searching the FAQ document."""
    chunks: List[str]
    query_used: str


class ComparisonResult(BaseModel):
    """What ComparisonAgent returns when asked to compare two periods."""
    period_a_label: str
    period_b_label: str
    period_a: DataResult
    period_b: DataResult
    notable_changes: List[str]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    """The full response sent back from the /ask endpoint."""
    question: str
    answer: str
    task_spec: TaskSpec
    data_result: Optional[DataResult] = None
    rag_result: Optional[RagResult] = None
    comparison_result: Optional[ComparisonResult] = None
