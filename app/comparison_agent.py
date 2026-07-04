"""
comparison_agent.py

ComparisonAgent answers "how does X compare to Y" questions. It does not
have its own metric logic - it just runs DataAgent twice (once per period)
and then looks at the difference between the two results.

Right now it always compares "this month" vs "last month" since that
covers the assignment's example question. A more complete version would
let the orchestrator pass in arbitrary date ranges parsed from the
question, which is called out as a next step in the README.
"""

from app.data_agent import filter_last_month, filter_this_month, run_data_agent
from app.models import ComparisonResult


def run_comparison_agent(df) -> ComparisonResult:
    this_month_df = filter_this_month(df)
    last_month_df = filter_last_month(df)

    this_month_result = run_data_agent(this_month_df)
    last_month_result = run_data_agent(last_month_df)

    changes = []

    csat_diff = round(this_month_result.csat - last_month_result.csat, 1)
    if abs(csat_diff) >= 1:
        direction = "up" if csat_diff > 0 else "down"
        changes.append(f"CSAT is {direction} {abs(csat_diff)} points versus last month")

    rating_diff = round(this_month_result.average_rating - last_month_result.average_rating, 2)
    if abs(rating_diff) >= 0.1:
        direction = "up" if rating_diff > 0 else "down"
        changes.append(f"Average rating is {direction} {abs(rating_diff)} versus last month")

    this_month_top_theme_names = {t.theme for t in this_month_result.top_themes}
    last_month_top_theme_names = {t.theme for t in last_month_result.top_themes}
    new_themes = this_month_top_theme_names - last_month_top_theme_names
    for theme in new_themes:
        changes.append(f"'{theme}' is a new top theme this month that wasn't in last month's top themes")

    if not changes:
        changes.append("no major changes between the two periods")

    return ComparisonResult(
        period_a_label="this month",
        period_b_label="last month",
        period_a=this_month_result,
        period_b=last_month_result,
        notable_changes=changes,
    )
