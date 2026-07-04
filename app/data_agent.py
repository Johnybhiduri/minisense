"""
data_agent.py

DataAgent's job is simple: take the survey data and turn it into real
numbers - average rating, CSAT, how many responses, and which themes show
up most in the free text complaints/praise.

We use plain pandas for this instead of asking an LLM to "guess" the
numbers. Math should come from math, not from a language model - an LLM
is much better used later to explain what the numbers mean.

The assignment asks for at least one example of tool calling from within
an agent, so compute_csat below is wrapped as a proper LangChain @tool.
We call it directly here rather than making an LLM decide to call it,
because for something this deterministic that would just add cost and a
chance of the LLM getting the math wrong. The @tool wrapper still means
this same function could be handed to a tool-calling LLM later with zero
changes.
"""

from datetime import datetime
from typing import List

import pandas as pd
from langchain_core.tools import tool

from app.models import DataResult, ThemeCount

# Same theme keywords used by the data generator, so we can spot them again
# in the free text. In a real system this list would come from whatever
# taxonomy the business already uses, or from a first clustering pass.
THEME_KEYWORDS = {
    "wait_time": ["wait", "line", "slow", "quick", "fast"],
    "food_quality": ["food", "quality", "fresh", "undercooked", "cold"],
    "staff": ["staff", "team member", "rude", "friendly", "helpful"],
    "price": ["price", "priced", "value", "worth", "expensive"],
    "cleanliness": ["clean", "dirty", "spotless", "restroom", "maintained"],
}


def load_survey_data(path: str) -> pd.DataFrame:
    """Load the survey JSON file into a pandas DataFrame."""
    df = pd.read_json(path)
    # the JSON has one top level key "responses" holding the list
    df = pd.json_normalize(df["responses"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@tool
def compute_csat(ratings: List[int]) -> float:
    """
    Compute CSAT (customer satisfaction) as the percentage of ratings
    that are 4 or 5 out of 5. Returns a number like 82.3 (percent).
    """
    if not ratings:
        return 0.0
    satisfied = sum(1 for r in ratings if r >= 4)
    return round((satisfied / len(ratings)) * 100, 1)


def compute_average_rating(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return round(df["rating"].mean(), 2)


def compute_top_themes(df: pd.DataFrame, top_n: int = 3) -> List[ThemeCount]:
    """
    Count how many free_text responses mention each theme's keywords, then
    return the top N themes by count. A response can count toward more
    than one theme if it mentions more than one thing.
    """
    total = len(df)
    if total == 0:
        return []

    counts = {}
    lowered_text = df["free_text"].str.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        mask = lowered_text.apply(lambda text: any(word in text for word in keywords))
        counts[theme] = int(mask.sum())

    sorted_themes = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_themes = sorted_themes[:top_n]

    return [
        ThemeCount(
            theme=theme,
            count=count,
            percentage=round((count / total) * 100, 1),
        )
        for theme, count in top_themes
    ]


def run_data_agent(df: pd.DataFrame) -> DataResult:
    """
    Main entry point for DataAgent. Takes an already-filtered DataFrame
    (the orchestrator decides the date range) and returns a DataResult.
    """
    if df.empty:
        date_range = "no data"
    else:
        start = df["date"].min().strftime("%Y-%m-%d")
        end = df["date"].max().strftime("%Y-%m-%d")
        date_range = f"{start} to {end}"

    ratings = df["rating"].tolist()

    return DataResult(
        response_count=len(df),
        average_rating=compute_average_rating(df),
        csat=compute_csat.invoke({"ratings": ratings}),
        top_themes=compute_top_themes(df),
        date_range=date_range,
    )


def filter_last_n_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep only rows from the last N days, based on today's date."""
    cutoff = datetime.now() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff]


def filter_this_month(df: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now()
    return df[(df["date"].dt.year == now.year) & (df["date"].dt.month == now.month)]


def filter_last_month(df: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now()
    # roll back to the 1st of this month, then step back one day to land
    # in the previous month
    first_of_this_month = now.replace(day=1)
    last_month_date = first_of_this_month - pd.Timedelta(days=1)
    return df[
        (df["date"].dt.year == last_month_date.year)
        & (df["date"].dt.month == last_month_date.month)
    ]
