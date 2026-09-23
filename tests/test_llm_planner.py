import json

import polars as pl
import pytest

from data_analyst.llm_planner import LLMPlanningError, OpenAIPlanner


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        payload = self.payload

        class Response:
            output_text = json.dumps(payload)

        return Response()


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def schema_df():
    return pl.DataFrame(
        {
            "category": ["a", "b", "a"],
            "revenue": [10.0, 20.0, 30.0],
            "orders": [1, 2, 3],
            "created_at": pl.datetime_range(
                pl.datetime(2026, 1, 1),
                pl.datetime(2026, 1, 3),
                interval="1d",
                eager=True,
            ),
        }
    )


def planner_for(df, payload):
    return OpenAIPlanner(client=FakeClient(payload))(
        "test question",
        columns=df.columns,
        numeric_columns=[name for name, dtype in df.schema.items() if dtype.is_numeric()],
        datetime_columns=[name for name, dtype in df.schema.items() if dtype in (pl.Date, pl.Datetime)],
    )


def test_accepts_safe_structured_plan():
    plan = planner_for(schema_df(), {
        "operation": "aggregate", "columns": [], "group_by": ["category"],
        "metric": "revenue", "aggregation": "mean", "datetime_column": None,
        "frequency": None, "sort_direction": "desc", "limit": 5, "assumptions": [],
        "filter_column": None, "filter_value": None,
    })
    assert plan.operation == "aggregate"
    assert plan.group_by == ["category"]
    assert plan.metric == "revenue"


def test_rejects_unknown_column():
    with pytest.raises(LLMPlanningError, match="unknown column"):
        planner_for(schema_df(), {
            "operation": "describe", "columns": ["not_a_column"], "group_by": [],
            "metric": None, "aggregation": None, "datetime_column": None,
            "frequency": None, "sort_direction": None, "limit": None, "assumptions": [],
        })


def test_rejects_invalid_operation():
    with pytest.raises(LLMPlanningError, match="Unsupported operation"):
        planner_for(schema_df(), {
            "operation": "sql", "columns": [], "group_by": [], "metric": None,
            "aggregation": None, "datetime_column": None, "frequency": None,
            "sort_direction": None, "limit": None, "assumptions": [],
        })


def test_rejects_non_numeric_metric():
    with pytest.raises(LLMPlanningError, match="must be numeric"):
        planner_for(schema_df(), {
            "operation": "aggregate", "columns": [], "group_by": ["category"],
            "metric": "category", "aggregation": "mean", "datetime_column": None,
            "frequency": None, "sort_direction": None, "limit": None, "assumptions": [],
        })


def test_requires_api_key_only_for_live_client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    planner = OpenAIPlanner()
    with pytest.raises(LLMPlanningError, match="OPENAI_API_KEY"):
        planner("describe revenue", columns=["revenue"], numeric_columns=["revenue"], datetime_columns=[])


def test_rejects_non_numeric_correlation_column():
    with pytest.raises(LLMPlanningError, match="Correlation columns must all be numeric"):
        planner_for(schema_df(), {
            "operation": "correlation", "columns": ["revenue", "category"], "group_by": [],
            "metric": None, "aggregation": None, "datetime_column": None,
            "frequency": None, "sort_direction": None, "limit": None, "assumptions": [],
        })


def test_rejects_duplicate_group_columns():
    with pytest.raises(LLMPlanningError, match="duplicate"):
        planner_for(schema_df(), {
            "operation": "aggregate", "columns": [], "group_by": ["category", "category"],
            "metric": "revenue", "aggregation": "mean", "datetime_column": None,
            "frequency": None, "sort_direction": "desc", "limit": 5, "assumptions": [],
        })


def test_rejects_limit_above_safety_cap():
    with pytest.raises(LLMPlanningError, match="between 1 and 1000"):
        planner_for(schema_df(), {
            "operation": "aggregate", "columns": [], "group_by": ["category"],
            "metric": "revenue", "aggregation": "mean", "datetime_column": None,
            "frequency": None, "sort_direction": "desc", "limit": 1001, "assumptions": [],
        })


def test_requires_complete_time_series_plan():
    with pytest.raises(LLMPlanningError, match="requires a frequency"):
        planner_for(schema_df(), {
            "operation": "time_series", "columns": [], "group_by": [],
            "metric": "revenue", "aggregation": "mean", "datetime_column": "created_at",
            "frequency": None, "sort_direction": None, "limit": None, "assumptions": [],
        })
