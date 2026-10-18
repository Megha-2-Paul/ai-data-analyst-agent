import numpy as np
import polars as pl
import pytest

from data_analyst.agent import AnalystAgent
from data_analyst.answer import generate_answer
from data_analyst.ml import MLAnalysisError, detect_anomalies, train_predictive_model


def test_anomaly_detection_returns_traceable_flags_without_mutating_input():
    values = [float(i) for i in range(29)] + [1000.0]
    frame = pl.DataFrame({"value": values, "second": [v * 2 for v in values]})
    original_columns = frame.columns.copy()

    result = detect_anomalies(frame, ["value"], contamination=0.1, random_state=7)

    assert result["analysis"] == "anomaly_detection"
    assert result["method"] == "IsolationForest"
    assert result["metadata"]["rows_scored"] == 30
    assert result["metadata"]["anomaly_count"] >= 1
    assert result["result"][0]["row_index"] == 29
    assert frame.columns == original_columns


def test_anomaly_detection_rejects_small_datasets():
    with pytest.raises(MLAnalysisError, match="at least 10 rows"):
        detect_anomalies(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))


def test_anomaly_detection_rejects_non_numeric_columns():
    frame = pl.DataFrame({"category": ["a"] * 12})
    with pytest.raises(MLAnalysisError, match="requires at least one numeric column"):
        detect_anomalies(frame)


def test_agent_routes_anomaly_question_to_ml_module():
    frame = pl.DataFrame({"sales": [float(i) for i in range(29)] + [1000.0]})
    response = AnalystAgent().ask("Find outliers in sales", frame)

    assert response.plan.operation == "anomaly_detection"
    assert response.result["analysis"] == "anomaly_detection"
    assert "Isolation Forest" in response.plan.assumptions[0]
    assert response.answer is not None
    assert response.answer.evidence


def test_regression_model_reports_held_out_metrics():
    frame = pl.DataFrame({
        "feature": [float(i) for i in range(50)],
        "category": ["A" if i % 2 == 0 else "B" for i in range(50)],
        "target": [float(3 * i + (5 if i % 2 else 0)) for i in range(50)],
    })
    result = train_predictive_model(
        frame, target="target", features=["feature", "category"],
        task="regression", test_size=0.2, random_state=42,
    )

    assert result["analysis"] == "predictive_model_evaluation"
    assert result["method"] == "RandomForestRegressor"
    assert result["result"]["train_rows"] == 40
    assert result["result"]["test_rows"] == 10
    assert set(result["result"]["metrics"]) == {"mae", "rmse", "r2"}
    assert np.isfinite(result["result"]["metrics"]["rmse"])
    assert result["metadata"]["limitations"]
    answer = generate_answer(result)
    assert answer.evidence
    assert "held-out" in answer.summary


def test_classification_model_reports_weighted_metrics():
    frame = pl.DataFrame({
        "feature": [float(i) for i in range(60)],
        "category": ["A" if i % 2 == 0 else "B" for i in range(60)],
        "target": ["yes" if i % 3 else "no" for i in range(60)],
    })
    result = train_predictive_model(
        frame, target="target", features=["feature", "category"],
        task="classification", random_state=11,
    )

    assert result["method"] == "RandomForestClassifier"
    assert set(result["result"]["metrics"]) == {
        "accuracy", "precision_weighted", "recall_weighted", "f1_weighted"
    }
    assert 0 <= result["result"]["metrics"]["accuracy"] <= 1


def test_predictive_model_prevents_target_leakage_by_argument_validation():
    frame = pl.DataFrame({"feature": [float(i) for i in range(25)], "target": [float(i) for i in range(25)]})
    with pytest.raises(MLAnalysisError, match="cannot also be a feature"):
        train_predictive_model(frame, target="target", features=["feature", "target"])


def test_predictive_model_requires_sufficient_data():
    frame = pl.DataFrame({"feature": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
    with pytest.raises(MLAnalysisError, match="at least 20 rows"):
        train_predictive_model(frame, target="target", features=["feature"])
