"""Provider adapters are tested offline with fake clients."""

import json

import pytest

from data_analyst.llm_planner import LLMPlanningError
from data_analyst.providers import GeminiPlanner, GroqPlanner

VALID = {
    "operation": "aggregate", "columns": [], "group_by": ["category"],
    "metric": "revenue", "aggregation": "mean", "datetime_column": None,
    "frequency": None, "sort_direction": "desc", "limit": 1, "assumptions": [],
    "filter_column": None, "filter_value": None,
}


class FakeGroq:
    def __init__(self, content):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        response = type("Response", (), {"choices": [choice]})()
        completions = type("Completions", (), {"create": lambda _self, **kwargs: response})()
        self.chat = type("Chat", (), {"completions": completions})()


class FakeGemini:
    def __init__(self, content):
        response = type("Response", (), {"text": content})()
        self.models = type("Models", (), {"generate_content": lambda _self, **kwargs: response})()


def plan_with(planner):
    return planner(
        "Which category has the highest average revenue?",
        columns=["category", "revenue"],
        numeric_columns=["revenue"],
        datetime_columns=[],
    )


def test_groq_adapter_validates_structured_response():
    plan = plan_with(GroqPlanner(client=FakeGroq(json.dumps(VALID))))
    assert plan.operation == "aggregate"
    assert plan.metric == "revenue"


def test_gemini_adapter_validates_structured_response():
    plan = plan_with(GeminiPlanner(client=FakeGemini(json.dumps(VALID))))
    assert plan.operation == "aggregate"
    assert plan.group_by == ["category"]


@pytest.mark.parametrize("planner_cls,key", [(GroqPlanner, "GROQ_API_KEY"), (GeminiPlanner, "GEMINI_API_KEY")])
def test_provider_requires_key_for_live_client(planner_cls, key, monkeypatch):
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(LLMPlanningError, match="not configured"):
        plan_with(planner_cls())


@pytest.mark.parametrize("planner_cls", [GroqPlanner, GeminiPlanner])
def test_provider_rejects_invalid_model_output(planner_cls):
    client = FakeGroq('{"operation":"execute_python"}') if planner_cls is GroqPlanner else FakeGemini('{"operation":"execute_python"}')
    with pytest.raises(LLMPlanningError, match="Unsupported operation"):
        plan_with(planner_cls(client=client))
