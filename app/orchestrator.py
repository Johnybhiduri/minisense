"""
orchestrator.py

The orchestrator is the "planner" - it looks at the incoming question,
decides which sub-agents need to run, and then combines their results
into one final answer.

A note on how the planning works: instead of using an LLM to decide the
plan, we use a small set of keyword rules (does the question mention
"compare", "vs", "last month", and so on). This is a deliberate choice for
a project this size - it's instant, free, and 100% predictable, versus an
LLM call that adds latency and could occasionally misroute a question.
The trade-off is that it won't handle every possible phrasing. The README
explains how this would be swapped for an LLM-based planner (using
with_structured_output against the TaskSpec model) if the question types
became more varied.

Either way, the important part the assignment asks for still holds: the
orchestrator builds one structured TaskSpec and hands it to each sub-agent,
rather than passing around raw question text.
"""

from app.comparison_agent import run_comparison_agent
from app.data_agent import load_survey_data, run_data_agent
from app.models import AskResponse, SubTask, TaskSpec
from app.rag_agent import run_rag_agent
from app.summary_agent import run_summary_agent

COMPARISON_KEYWORDS = ["compare", "vs", "versus", "last month", "this month", "change", "trend"]


def build_task_spec(question: str) -> TaskSpec:
    """Decide which sub-agents this question needs."""
    lowered = question.lower()
    subtasks = [
        SubTask(agent="data", instruction="Compute overall survey metrics for the relevant period."),
        SubTask(agent="rag", instruction=f"Find FAQ context relevant to: {question}"),
    ]

    if any(keyword in lowered for keyword in COMPARISON_KEYWORDS):
        subtasks.append(SubTask(agent="comparison", instruction="Compare this month against last month."))

    return TaskSpec(question=question, subtasks=subtasks)


def run_pipeline(question: str, survey_data_path: str, vector_store) -> AskResponse:
    """
    Main entry point used by the FastAPI app. Runs the full
    orchestrator -> sub-agents -> summary flow for one question.
    """
    task_spec = build_task_spec(question)
    df = load_survey_data(survey_data_path)

    data_result = None
    rag_result = None
    comparison_result = None

    for subtask in task_spec.subtasks:
        if subtask.agent == "data":
            data_result = run_data_agent(df)
        elif subtask.agent == "rag":
            rag_result = run_rag_agent(vector_store, subtask.instruction)
        elif subtask.agent == "comparison":
            comparison_result = run_comparison_agent(df)

    final_answer = run_summary_agent(
        question=question,
        data_result=data_result,
        rag_result=rag_result,
        comparison_result=comparison_result,
    )

    return AskResponse(
        question=question,
        answer=final_answer,
        task_spec=task_spec,
        data_result=data_result,
        rag_result=rag_result,
        comparison_result=comparison_result,
    )
