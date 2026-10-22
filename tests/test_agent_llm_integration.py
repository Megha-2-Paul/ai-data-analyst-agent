import json

import polars as pl

from data_analyst.agent import AnalystAgent
from data_analyst.llm_planner import OpenAIPlanner


class FakeResponses:
    def create(self, **kwargs):
        class Response:
            output_text = json.dumps({
                "operation": "aggregate", "columns": [], "group_by": ["category"],
                "metric": "revenue", "aggregation": "mean", "datetime_column": None,
                "frequency": None, "sort_direction": "desc", "limit": 1, "assumptions": [],
            })
        return Response()


class FakeClient:
    responses = FakeResponses()


def test_agent_can_use_llm_planner_without_network():
    df = pl.DataFrame({"category": ["a", "b", "a"], "revenue": [10.0, 20.0, 30.0]})
    response = AnalystAgent(planner=OpenAIPlanner(client=FakeClient())).ask(
        "Which category has the highest average revenue?", df
    )
    assert response.plan.operation == "aggregate"
    assert response.result["result"][0]["revenue"] == 20.0
