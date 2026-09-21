import polars as pl
import pytest

from data_analyst.agent import AnalystAgent
from data_analyst.reasoning import (
    AnalysisPlan,
    AnalysisStep,
    ReasoningError,
    execute_analysis_plan,
    plan_analysis,
)
from data_analyst.planner import QueryPlan


def make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "payment_type": ["card", "cash", "card", "cash", "card", "cash"],
            "fare_amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "pickup_datetime": pl.datetime_range(
                pl.datetime(2025, 1, 1),
                pl.datetime(2025, 1, 6),
                interval="1d",
                eager=True,
            ),
        }
    )


def test_plans_identify_then_time_series():
    plan = plan_analysis(
        "Which payment_type has the highest average fare_amount, and did that payment_type's average fare_amount increase over the year?",
        columns=["payment_type", "fare_amount", "pickup_datetime"],
        numeric_columns=["fare_amount"],
        datetime_columns=["pickup_datetime"],
    )
    assert [step.step_id for step in plan.steps] == ["step_1", "step_2"]
    assert plan.steps[0].plan.limit == 1
    assert plan.steps[0].plan.sort_direction == "desc"
    assert plan.steps[1].depends_on == ["step_1"]
    assert plan.final_step_id == "step_2"


def test_agent_executes_multi_step_reasoning():
    response = AnalystAgent().ask(
        "Which payment_type has the highest average fare_amount, and did that payment_type's average fare_amount increase over the year?",
        make_df(),
    )
    assert response.result["analysis"] == "multi_step_reasoning"
    assert response.result["final_step_id"] == "step_2"
    assert response.result["steps"][1]["filter"]["value"] == "cash"
    assert response.result["final_result"]["analysis"] == "time_series"
    assert response.execution_steps[-1] == "return_structured_result"


def test_rejects_unknown_dependency():
    step = AnalysisStep(
        step_id="step_2",
        plan=QueryPlan(operation="quality"),
        depends_on=["missing"],
    )
    with pytest.raises(ReasoningError, match="unknown dependencies"):
        execute_analysis_plan(
            make_df(),
            AnalysisPlan(question="x", steps=[step], final_step_id="step_2"),
            lambda df, plan: {},
        )


def test_rejects_circular_dependency():
    p = QueryPlan(operation="quality")
    steps = [
        AnalysisStep(step_id="step_1", plan=p, depends_on=["step_2"]),
        AnalysisStep(step_id="step_2", plan=p, depends_on=["step_1"]),
    ]
    with pytest.raises(ReasoningError, match="circular"):
        execute_analysis_plan(
            make_df(),
            AnalysisPlan(question="x", steps=steps, final_step_id="step_2"),
            lambda df, plan: {},
        )


def test_rejects_invalid_filter_binding():
    p = QueryPlan(operation="quality")
    steps = [
        AnalysisStep(step_id="step_1", plan=p),
        AnalysisStep(
            step_id="step_2",
            plan=p,
            depends_on=[],
            filter_from_step="step_1",
            filter_source_column="payment_type",
            filter_target_column="payment_type",
        ),
    ]
    with pytest.raises(ReasoningError, match="must be a dependency"):
        execute_analysis_plan(
            make_df(),
            AnalysisPlan(question="x", steps=steps, final_step_id="step_2"),
            lambda df, plan: {},
        )


def test_rejects_step_limit():
    p = QueryPlan(operation="quality")
    steps = [AnalysisStep(step_id=f"step_{i}", plan=p) for i in range(1, 10)]
    with pytest.raises(ReasoningError, match="more than 8"):
        execute_analysis_plan(
            make_df(),
            AnalysisPlan(question="x", steps=steps, final_step_id="step_9"),
            lambda df, plan: {},
        )


def test_single_step_questions_still_use_stage_1():
    response = AnalystAgent().ask(
        "What is the average fare_amount by payment_type?",
        make_df(),
    )
    assert response.plan.operation == "aggregate"
    assert response.result["analysis"] == "grouped_aggregation"
