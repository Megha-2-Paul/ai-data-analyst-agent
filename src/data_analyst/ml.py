"""Optional machine-learning tools for tabular analysis.

The deterministic analytics engine remains the source of truth for ordinary
aggregations. These tools are opt-in and return model metadata and limitations
alongside results. Install with: pip install -e ".[ml]".
"""
from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import polars as pl


class MLAnalysisError(ValueError):
    """Raised when a requested ML analysis is unsuitable for the supplied data."""


def _sklearn():
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
            precision_score, r2_score, recall_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            'Machine-learning features require the optional dependency. Install with: pip install -e ".[ml]"'
        ) from exc
    return {
        "pd": pd, "ColumnTransformer": ColumnTransformer, "IsolationForest": IsolationForest,
        "RandomForestClassifier": RandomForestClassifier, "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer, "accuracy_score": accuracy_score, "f1_score": f1_score,
        "mean_absolute_error": mean_absolute_error, "mean_squared_error": mean_squared_error,
        "precision_score": precision_score, "r2_score": r2_score, "recall_score": recall_score,
        "train_test_split": train_test_split, "Pipeline": Pipeline, "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def detect_anomalies(
    df: pl.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    contamination: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """Flag unusual rows with Isolation Forest; never modify the source dataframe."""
    lib = _sklearn()
    if not 0.0 < contamination <= 0.5:
        raise MLAnalysisError("contamination must be greater than 0 and at most 0.5.")
    selected = list(columns) if columns is not None else [
        name for name, dtype in df.schema.items() if dtype.is_numeric()
    ]
    if not selected:
        raise MLAnalysisError("Anomaly detection requires at least one numeric column.")
    missing = [name for name in selected if name not in df.columns]
    if missing:
        raise MLAnalysisError(f"Unknown column(s): {', '.join(missing)}.")
    non_numeric = [name for name in selected if not df.schema[name].is_numeric()]
    if non_numeric:
        raise MLAnalysisError(f"Anomaly detection requires numeric columns: {', '.join(non_numeric)}.")
    if df.height < 10:
        raise MLAnalysisError("Anomaly detection requires at least 10 rows.")
    numeric = df.select(selected).cast({name: pl.Float64 for name in selected}, strict=False)
    matrix = numeric.to_numpy()
    valid_mask = np.isfinite(matrix).all(axis=1)
    valid_positions = np.flatnonzero(valid_mask)
    if len(valid_positions) < 10:
        raise MLAnalysisError("At least 10 rows with finite values in every selected column are required.")
    model = lib["IsolationForest"](
        contamination=contamination, random_state=random_state, n_estimators=200
    )
    model.fit(matrix[valid_mask])
    labels = model.predict(matrix[valid_mask])
    scores = model.decision_function(matrix[valid_mask])
    flagged = [
        {"row_index": int(valid_positions[i]), "anomaly_score": float(-scores[i]),
         "values": {column: float(matrix[valid_positions[i], j]) for j, column in enumerate(selected)}}
        for i, label in enumerate(labels) if label == -1
    ]
    flagged.sort(key=lambda item: item["anomaly_score"], reverse=True)
    return {
        "analysis": "anomaly_detection",
        "method": "IsolationForest",
        "parameters": {"columns": selected, "contamination": contamination, "random_state": random_state},
        "result": flagged,
        "metadata": {
            "row_count": df.height, "rows_scored": int(len(valid_positions)),
            "rows_skipped_missing_or_nonfinite": int(df.height - len(valid_positions)),
            "anomaly_count": len(flagged),
            "limitations": [
                "Anomaly scores identify statistical unusualness, not errors, fraud, or business risk.",
                "Row indices refer to zero-based positions in the supplied dataframe.",
                "Results depend on selected columns, contamination, and the data distribution.",
            ],
        },
    }


def train_predictive_model(
    df: pl.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    task: Literal["regression", "classification"] = "regression",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a baseline random-forest model with a held-out test split."""
    lib = _sklearn()
    if task not in {"regression", "classification"}:
        raise MLAnalysisError("task must be 'regression' or 'classification'.")
    if not 0.1 <= test_size <= 0.4:
        raise MLAnalysisError("test_size must be between 0.1 and 0.4.")
    feature_list = list(dict.fromkeys(features))
    if not feature_list:
        raise MLAnalysisError("At least one feature column is required.")
    if target in feature_list:
        raise MLAnalysisError("The target column cannot also be a feature.")
    unknown = [name for name in [target, *feature_list] if name not in df.columns]
    if unknown:
        raise MLAnalysisError(f"Unknown column(s): {', '.join(unknown)}.")
    if df.height < 20:
        raise MLAnalysisError("Predictive modelling requires at least 20 rows.")
    subset = df.select([target, *feature_list]).drop_nulls(subset=[target])
    if subset.height < 20:
        raise MLAnalysisError("At least 20 non-null target rows are required.")
    y_series = subset.get_column(target)
    if task == "regression":
        if not y_series.dtype.is_numeric():
            raise MLAnalysisError("Regression requires a numeric target column.")
        y = y_series.cast(pl.Float64).to_numpy()
        finite = np.isfinite(y)
        subset = subset.filter(pl.Series(finite))
        y = y[finite]
        if len(y) < 20:
            raise MLAnalysisError("At least 20 finite numeric target rows are required.")
    else:
        y = y_series.cast(pl.String).to_numpy()
        if len(np.unique(y)) < 2:
            raise MLAnalysisError("Classification requires at least two target classes.")
    X = lib["pd"].DataFrame(subset.select(feature_list).to_dict(as_series=False))
    # Convert dates to stable numeric timestamps rather than silently treating them as categories.
    for name in feature_list:
        if X[name].dtype.kind == "M":
            X[name] = X[name].astype("int64") / 1_000_000_000
    numeric = [name for name in feature_list if X[name].dtype.kind in "biuf"]
    for name in numeric:
        X[name] = X[name].replace([np.inf, -np.inf], np.nan)
    categorical = [name for name in feature_list if name not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", lib["Pipeline"]([
            ("imputer", lib["SimpleImputer"](strategy="median")),
            ("scale", lib["StandardScaler"]()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", lib["Pipeline"]([
            ("imputer", lib["SimpleImputer"](strategy="most_frequent")),
            ("encode", lib["OneHotEncoder"](handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise MLAnalysisError("No usable feature columns remain after preprocessing.")
    preprocessor = lib["ColumnTransformer"](transformers)
    stratify = y if task == "classification" and min(np.unique(y, return_counts=True)[1]) >= 2 else None
    if stratify is not None and int(np.ceil(len(y) * test_size)) < len(np.unique(y)):
        raise MLAnalysisError("test_size is too small to include every class in the held-out split.")
    X_train, X_test, y_train, y_test = lib["train_test_split"](
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    model = (
        lib["RandomForestRegressor"](n_estimators=200, random_state=random_state, n_jobs=1)
        if task == "regression"
        else lib["RandomForestClassifier"](n_estimators=200, random_state=random_state, n_jobs=1, class_weight="balanced")
    )
    pipeline = lib["Pipeline"]([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    if task == "regression":
        metrics = {
            "mae": float(lib["mean_absolute_error"](y_test, predictions)),
            "rmse": float(np.sqrt(lib["mean_squared_error"](y_test, predictions))),
            "r2": float(lib["r2_score"](y_test, predictions)) if len(y_test) > 1 else None,
        }
    else:
        metrics = {
            "accuracy": float(lib["accuracy_score"](y_test, predictions)),
            "precision_weighted": float(lib["precision_score"](y_test, predictions, average="weighted", zero_division=0)),
            "recall_weighted": float(lib["recall_score"](y_test, predictions, average="weighted", zero_division=0)),
            "f1_weighted": float(lib["f1_score"](y_test, predictions, average="weighted", zero_division=0)),
        }
    return {
        "analysis": "predictive_model_evaluation",
        "method": "RandomForestRegressor" if task == "regression" else "RandomForestClassifier",
        "parameters": {"target": target, "features": feature_list, "task": task, "test_size": test_size, "random_state": random_state},
        "result": {"metrics": metrics, "test_rows": int(len(y_test)), "train_rows": int(len(y_train))},
        "metadata": {
            "target": target, "feature_count": len(feature_list), "rows_used": int(len(y)),
            "limitations": [
                "This is a baseline model, not evidence of causal relationships.",
                "Review target leakage and feature availability at prediction time before relying on the model.",
                "Metrics are based on one random holdout split and may vary with the data and random seed.",
            ],
        },
    }



def forecast_time_series(
    df: pl.DataFrame,
    *,
    datetime_column: str,
    target: str,
    horizon: int = 1,
    test_size: float = 0.2,
    n_lags: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Forecast a numeric series with lag features and a chronological holdout.

    This is a baseline forecast, not a causal model. The final observations are
    reserved for evaluation; random shuffling is deliberately not used.
    """
    lib = _sklearn()
    if datetime_column not in df.columns or target not in df.columns:
        missing = [name for name in (datetime_column, target) if name not in df.columns]
        raise MLAnalysisError(f"Unknown column(s): {', '.join(missing)}.")
    dtype = df.schema[datetime_column]
    if dtype not in (pl.Date, pl.Datetime, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        raise MLAnalysisError("Forecasting requires a Date/Datetime column or an integer year column.")
    if not df.schema[target].is_numeric():
        raise MLAnalysisError("Forecasting requires a numeric target column.")
    if not 1 <= horizon <= 24:
        raise MLAnalysisError("horizon must be between 1 and 24 steps.")
    if not 0.1 <= test_size <= 0.4:
        raise MLAnalysisError("test_size must be between 0.1 and 0.4.")
    if not 1 <= n_lags <= 12:
        raise MLAnalysisError("n_lags must be between 1 and 12.")
    series_frame = df.select([datetime_column, target]).drop_nulls().sort(datetime_column)
    raw_values = series_frame.get_column(target).cast(pl.Float64).to_numpy()
    finite = np.isfinite(raw_values)
    series_frame = series_frame.filter(pl.Series(finite))
    raw_values = raw_values[finite]
    if len(raw_values) < max(20, n_lags + 12):
        raise MLAnalysisError(f"Forecasting requires at least {max(20, n_lags + 12)} finite observations.")
    X = np.asarray([
        [raw_values[index - lag] for lag in range(1, n_lags + 1)]
        for index in range(n_lags, len(raw_values))
    ], dtype=float)
    # A monotonic time index helps the model distinguish recurring lag values
    # at different points in the series without assuming a date frequency.
    time_index = np.arange(n_lags, len(raw_values), dtype=float).reshape(-1, 1)
    X = np.hstack([X, time_index])
    y = raw_values[n_lags:]
    test_count = max(2, int(np.ceil(len(y) * test_size)))
    train_count = len(y) - test_count
    if train_count < 10:
        raise MLAnalysisError("Not enough training observations after reserving the chronological test window.")
    X_train, X_test = X[:train_count], X[train_count:]
    y_train, y_test = y[:train_count], y[train_count:]
    model = lib["RandomForestRegressor"](n_estimators=200, random_state=random_state, n_jobs=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    naive_predictions = X_test[:, 0]
    metrics = {
        "mae": float(lib["mean_absolute_error"](y_test, predictions)),
        "rmse": float(np.sqrt(lib["mean_squared_error"](y_test, predictions))),
        "naive_mae": float(lib["mean_absolute_error"](y_test, naive_predictions)),
        "naive_rmse": float(np.sqrt(lib["mean_squared_error"](y_test, naive_predictions))),
        "r2": float(lib["r2_score"](y_test, predictions)) if len(y_test) > 1 else None,
    }
    history = list(raw_values)
    future = []
    for step in range(1, horizon + 1):
        features = [history[-lag] for lag in range(1, n_lags + 1)] + [float(len(history))]
        prediction = float(model.predict(np.asarray(features, dtype=float).reshape(1, -1))[0])
        future.append({"step_ahead": step, "predicted_value": prediction})
        history.append(prediction)
    return {
        "analysis": "time_series_forecast",
        "method": "RandomForestRegressor",
        "parameters": {
            "datetime_column": datetime_column, "target": target, "horizon": horizon,
            "test_size": test_size, "n_lags": n_lags, "random_state": random_state,
        },
        "result": {
            "metrics": metrics, "train_rows": int(train_count), "test_rows": int(test_count),
            "future_forecast": future,
        },
        "metadata": {
            "observations_used": int(len(raw_values)),
            "last_observed_time": str(series_frame.get_column(datetime_column)[-1]),
            "evaluation": "chronological holdout; final observations were not used for fitting",
            "limitations": [
                "Forecast steps are relative to the last observation; no calendar frequency is inferred.",
                "Compare model MAE/RMSE with the naive baseline before using the forecast.",
                "Forecasts are extrapolations, not causal explanations or guarantees.",
            ],
        },
    }
