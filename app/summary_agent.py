"""
summary_agent.py

SummaryAgent turns the structured metrics + retrieved FAQ context into one
narrative paragraph. Instead of downloading and running a model locally
(this used to pull google/flan-t5-base, about 1GB), it now calls Hugging
Face's free Inference API - no big download, just an API call. The free
tier is rate-limited, but that's a fine trade-off for a demo like this.

You'll need a free Hugging Face account and access token for this to work:
1. Sign up at https://huggingface.co/join (free)
2. Create a token at https://huggingface.co/settings/tokens (read access is enough)
3. Put it in your .env file as HUGGINGFACEHUB_API_TOKEN=your_token_here

If the API call fails for any reason (rate limit, missing token, no
internet), we fall back to a simple template-based paragraph instead of
crashing the whole app.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.models import ComparisonResult, DataResult, RagResult

load_dotenv()  # reads HUGGINGFACEHUB_API_TOKEN (and anything else) from a local .env file

# You can swap this for any instruct model available on Hugging Face's free
# Inference API if this one is rate limited or unavailable for you - just
# set HF_SUMMARY_MODEL in your .env file to override it.
SUMMARY_MODEL = os.getenv("HF_SUMMARY_MODEL", "Qwen/Qwen2.5-7B-Instruct")

_chat_model = None  # created once, on first use


def _get_chat_model():
    global _chat_model
    if _chat_model is None:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        llm = HuggingFaceEndpoint(
            repo_id=SUMMARY_MODEL,
            task="text-generation",
            max_new_tokens=200,
            temperature=0.3,
            provider="auto",  # let Hugging Face route to whichever provider is available
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        )
        _chat_model = ChatHuggingFace(llm=llm)
    return _chat_model


def _build_prompt(
    question: str,
    data_result: Optional[DataResult],
    rag_result: Optional[RagResult],
    comparison_result: Optional[ComparisonResult],
) -> str:
    """Put all the structured pieces into one plain-text prompt for the model."""
    parts = [
        f"Question: {question}",
        "Write a short, clear paragraph answering this for a business owner, using only the facts below.",
    ]

    if data_result:
        parts.append(
            f"Survey facts: {data_result.response_count} responses, "
            f"average rating {data_result.average_rating} out of 5, "
            f"CSAT {data_result.csat} percent, "
            f"top themes: {', '.join(t.theme for t in data_result.top_themes)}."
        )

    if comparison_result:
        parts.append("Comparison facts: " + "; ".join(comparison_result.notable_changes) + ".")

    if rag_result and rag_result.chunks:
        context = " ".join(rag_result.chunks)[:600]
        parts.append(f"Company background: {context}")

    return "\n".join(parts)


def _fallback_summary(
    data_result: Optional[DataResult],
    comparison_result: Optional[ComparisonResult],
) -> str:
    """A plain, template-based paragraph used if the API call isn't available."""
    pieces = []
    if data_result:
        theme_names = ", ".join(t.theme for t in data_result.top_themes) or "no clear theme"
        pieces.append(
            f"Based on {data_result.response_count} responses, the average rating is "
            f"{data_result.average_rating} out of 5 with a CSAT of {data_result.csat}%. "
            f"The most mentioned themes are: {theme_names}."
        )
    if comparison_result:
        pieces.append("Compared to last month: " + "; ".join(comparison_result.notable_changes) + ".")
    if not pieces:
        pieces.append("Not enough data was available to answer this question.")
    return " ".join(pieces)


def run_summary_agent(
    question: str,
    data_result: Optional[DataResult] = None,
    rag_result: Optional[RagResult] = None,
    comparison_result: Optional[ComparisonResult] = None,
) -> str:
    prompt = _build_prompt(question, data_result, rag_result, comparison_result)

    try:
        chat_model = _get_chat_model()
        response = chat_model.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
        if answer:
            return answer
    except Exception as error:
        print(f"[summary_agent] falling back to template answer, API call failed: {error}")

    return _fallback_summary(data_result, comparison_result)