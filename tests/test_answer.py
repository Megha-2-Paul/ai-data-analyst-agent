import polars as pl
import pytest

from data_analyst.answer import AnswerGenerationError, generate_answer, validate_answer
from data_analyst.agent import AnalystAgent
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


def test_generates_evidence_backed_multi_step_answer():
    response = AnalystAgent().ask(
        "Which payment_type has the highest average fare_amount, and did that payment_type's average fare_amount increase over the year?",
        make_df(),
    )
    answer = generate_answer(response.result)
    assert answer.findings
    assert len(answer.evidence) == 2
    assert all(f.evidence_ids for f in answer.findings)
    assert "cash" in answer.summary
    assert "increased" in answer.summary
    # The synthetic cash series goes from 20 to 60, a 200% increase.
    assert "200.0" in answer.summary or "200.00" in answer.summary


def test_generates_grouped_answer():
    response = AnalystAgent().ask("What is the average fare_amount by payment_type?", make_df())
    answer = generate_answer(response.result)
    assert "fare_amount by payment_type" in answer.findings[0].text
    assert len(answer.evidence) == 2
    assert [item.values["context"] for item in answer.evidence] == [
        "payment_type=card",
        "payment_type=cash",
    ]
    assert [item.values["value"] for item in answer.evidence] == [30.0, 40.0]


def test_grouped_answer_respects_ranking_intent():
    response = AnalystAgent().ask(
        "Which payment_type has the highest average fare_amount?",
        make_df(),
    )
    answer = generate_answer(response.result)
    assert "highest fare_amount" in answer.findings[0].text
    assert len(answer.evidence) == 1
    assert answer.evidence[0].values["context"] == "payment_type=cash"


def test_rejects_finding_without_evidence():
    with pytest.raises(AnswerGenerationError, match="no evidence"):
        validate_answer(
            __import__("data_analyst.answer", fromlist=["AnalystAnswer"]).AnalystAnswer(
                summary="bad",
                findings=[__import__("data_analyst.answer", fromlist=["Finding"]).Finding("f1", "bad", [])],
                evidence=[],
            )
        )


def test_agent_response_can_include_answer():
    response = AnalystAgent().ask("What is the average fare_amount by payment_type?", make_df())
    answer = generate_answer(response.result)
    payload = answer.to_dict()
    assert payload["findings"][0]["evidence_ids"] == ["e1"]
    assert payload["evidence"][0]["source_step_id"] == "step_1"


def test_unsupported_analysis_is_explicit():
    answer = generate_answer({"analysis": "unknown", "result": []})
    assert answer.findings == []
    assert answer.limitations
