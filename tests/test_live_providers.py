"""Opt-in live smoke tests; skipped unless provider API keys are configured.

These tests make real network requests and consume the provider account's quota.
"""

import os

import pytest

from data_analyst.providers import GeminiPlanner, GroqPlanner


@pytest.mark.parametrize(
    "key,planner_cls",
    [("GROQ_API_KEY", GroqPlanner), ("GEMINI_API_KEY", GeminiPlanner)],
)
def test_live_provider_returns_a_validated_plan(key, planner_cls):
    if not os.getenv(key):
        pytest.skip(f"{key} is not configured; live provider smoke test skipped.")
    plan = planner_cls()(
        "Which category has the highest average revenue?",
        columns=["category", "revenue"],
        numeric_columns=["revenue"],
        datetime_columns=[],
    )
    assert plan.operation in {"aggregate", "describe", "profile", "quality", "correlation", "time_series"}
