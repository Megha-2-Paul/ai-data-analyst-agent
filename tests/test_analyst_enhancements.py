import polars as pl
import pytest

from data_analyst.agent import AnalystAgent
from data_analyst.planner import PlanningError, plan_query


def test_global_aggregation_is_supported():
    frame = pl.DataFrame({"sales": [10.0, 20.0, 30.0]})
    response = AnalystAgent().ask("What is the average sales?", frame)

    assert response.plan.operation == "aggregate"
    assert response.plan.group_by == []
    assert response.result["result"] == [{"sales": 20.0}]
    assert response.answer is not None
    assert response.answer.evidence
    assert "20.00" in response.answer.summary


def test_explicit_filter_is_applied_before_global_aggregation():
    frame = pl.DataFrame({
        "status": ["active", "inactive", "active"],
        "sales": [10.0, 100.0, 30.0],
    })
    response = AnalystAgent().ask(
        "What is the average sales where status is active?", frame
    )

    assert response.plan.filter_column == "status"
    assert response.plan.filter_value.lower() == "active"
    assert response.result["result"] == [{"sales": 20.0}]


def test_multi_column_grouping_is_preserved():
    frame = pl.DataFrame({
        "region": ["East", "East", "West"],
        "product": ["A", "B", "A"],
        "sales": [10.0, 20.0, 30.0],
    })
    response = AnalystAgent().ask("What is the total sales by region and product?", frame)

    assert response.plan.group_by == ["region", "product"]
    assert len(response.result["result"]) == 3


def test_percentage_of_total_is_calculated_for_grouped_values():
    frame = pl.DataFrame({
        "region": ["East", "East", "West"],
        "sales": [10.0, 20.0, 70.0],
    })
    response = AnalystAgent().ask(
        "What percentage of total sales by region?", frame
    )

    assert response.plan.percentage_of_total is True
    rows = {row["region"]: row for row in response.result["result"]}
    assert rows["East"]["percentage_of_total"] == pytest.approx(30.0)
    assert rows["West"]["percentage_of_total"] == pytest.approx(70.0)
    assert response.result["metadata"]["percentage_denominator"] == 100.0


def test_descriptive_statistics_include_iqr_and_missing_rate():
    frame = pl.DataFrame({"sales": [10.0, 20.0, None, 40.0]})
    response = AnalystAgent().ask("Give descriptive statistics for sales", frame)
    sales = response.result["result"][0]

    assert sales["missing_rate"] == pytest.approx(0.25)
    assert sales["iqr"] == pytest.approx(15.0)
    assert "variance" in sales
