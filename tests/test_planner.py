import json
import sys

import polars as pl
import pytest

from data_analyst.analysis import aggregate, time_series
from data_analyst.cli import main
from data_analyst.planner import PlanningError, plan_query


COLUMNS = ["payment_type", "fare_amount", "trip_distance", "pickup_datetime"]
NUMERIC = ["fare_amount", "trip_distance"]
DATETIME = ["pickup_datetime"]


def test_plans_grouped_average():
    plan = plan_query(
        "What is the average fare_amount by payment_type?",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    assert plan.operation == "aggregate"
    assert plan.group_by == ["payment_type"]
    assert plan.metric == "fare_amount"
    assert plan.aggregation == "mean"


def test_plans_top_n_grouped_query():
    plan = plan_query(
        "Show the top 5 payment_type groups by highest average fare_amount",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    assert plan.sort_direction == "desc"
    assert plan.limit == 5


@pytest.mark.parametrize(
    ("question", "direction"),
    [
        ("Which payment_type has the highest average fare_amount?", "desc"),
        ("Which payment_type has the lowest average fare_amount?", "asc"),
        ("Which fare_amount has the highest average payment_type?", "desc"),
    ],
)
def test_plans_which_ranking_with_group_and_metric_roles(question, direction):
    # payment_type is deliberately included in numeric_columns to reproduce
    # schemas where a numeric-looking category can be mistaken for the metric.
    plan = plan_query(
        question,
        columns=COLUMNS,
        numeric_columns=["payment_type", "fare_amount", "trip_distance"],
        datetime_columns=DATETIME,
    )

    assert plan.operation == "aggregate"
    expected_group = "fare_amount" if "Which fare_amount" in question else "payment_type"
    expected_metric = "payment_type" if expected_group == "fare_amount" else "fare_amount"
    assert plan.group_by == [expected_group]
    assert plan.metric == expected_metric
    assert plan.sort_direction == direction
    assert plan.limit == 1


def test_explicit_top_n_ranking_keeps_requested_limit():
    plan = plan_query(
        "Show the top 5 payment_type groups by highest average fare_amount",
        columns=COLUMNS,
        numeric_columns=["payment_type", "fare_amount", "trip_distance"],
        datetime_columns=DATETIME,
    )

    assert plan.group_by == ["payment_type"]
    assert plan.metric == "fare_amount"
    assert plan.limit == 5


def test_plans_correlation():
    plan = plan_query(
        "What is the correlation between fare_amount and trip_distance?",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    assert plan.operation == "correlation"
    assert plan.columns == ["fare_amount", "trip_distance"]


def test_plans_time_series():
    plan = plan_query(
        "Show the daily trend of average fare_amount over pickup_datetime",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    assert plan.operation == "time_series"
    assert plan.datetime_column == "pickup_datetime"
    assert plan.metric == "fare_amount"
    assert plan.aggregation == "mean"
    assert plan.frequency == "1d"


def test_plans_descriptive_statistics():
    plan = plan_query(
        "Give me descriptive statistics for trip_distance",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    assert plan.operation == "describe"
    assert plan.columns == ["trip_distance"]


def test_plans_quality():
    plan = plan_query(
        "Run a data quality check",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    assert plan.operation == "quality"


def test_plans_profile_intent():
    plan = plan_query("Profile the dataset", columns=COLUMNS)

    assert plan.operation == "profile"


@pytest.mark.parametrize(
    "question",
    [
        "Give me descriptive statistics for payment_type",
        "What is the correlation between payment_type and pickup_datetime?",
    ],
)
def test_empty_numeric_schema_is_not_replaced_with_all_columns(question):
    with pytest.raises(PlanningError, match="numeric"):
        plan_query(
            question,
            columns=COLUMNS,
            numeric_columns=[],
            datetime_columns=[],
        )


def test_empty_datetime_schema_is_not_replaced_with_all_columns():
    with pytest.raises(PlanningError, match="date/datetime"):
        plan_query(
            "Show the daily trend over pickup_datetime",
            columns=COLUMNS,
            numeric_columns=[],
            datetime_columns=[],
        )


def test_rejects_multiple_candidate_metrics_for_one_grouped_aggregation():
    with pytest.raises(PlanningError, match="exactly one numeric metric"):
        plan_query(
            "What is the average fare_amount and trip_distance by payment_type?",
            columns=COLUMNS,
            numeric_columns=NUMERIC,
            datetime_columns=DATETIME,
        )


def test_rejects_multiple_candidate_datetime_columns():
    with pytest.raises(PlanningError, match="exactly one date/datetime column"):
        plan_query(
            "Show the daily trend over pickup_datetime and dropoff_datetime",
            columns=["fare_amount", "pickup_datetime", "dropoff_datetime"],
            numeric_columns=["fare_amount"],
            datetime_columns=["pickup_datetime", "dropoff_datetime"],
        )


def test_rejects_conflicting_time_series_frequencies():
    with pytest.raises(PlanningError, match="Conflicting time frequencies"):
        plan_query(
            "Show the daily and weekly trend over pickup_datetime",
            columns=COLUMNS,
            numeric_columns=NUMERIC,
            datetime_columns=DATETIME,
        )


def test_column_name_with_annual_does_not_conflict_with_daily_frequency():
    plan = plan_query(
        "Show daily trend of annual_revenue over pickup_datetime",
        columns=["annual_revenue", "pickup_datetime"],
        numeric_columns=["annual_revenue"],
        datetime_columns=["pickup_datetime"],
    )

    assert plan.operation == "time_series"
    assert plan.frequency == "1d"
    assert plan.metric == "annual_revenue"


def test_rejects_distinct_columns_with_same_normalized_match():
    with pytest.raises(PlanningError, match="Ambiguous column reference 'annual revenue'"):
        plan_query(
            "Show yearly average annual revenue over pickup_datetime",
            columns=["annual_revenue", "annual revenue", "pickup_datetime"],
            numeric_columns=["annual_revenue", "annual revenue"],
            datetime_columns=["pickup_datetime"],
        )


def test_rejects_unsupported_quarterly_frequency():
    with pytest.raises(PlanningError, match="Unsupported time frequency 'quarterly'"):
        plan_query(
            "Show quarterly trend of fare_amount over pickup_datetime",
            columns=COLUMNS,
            numeric_columns=NUMERIC,
            datetime_columns=DATETIME,
        )


def test_daily_time_series_default_is_recorded_as_an_assumption():
    plan = plan_query(
        "Show trend over pickup_datetime",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )

    assert plan.frequency == "1d"
    assert any("daily" in assumption.lower() for assumption in plan.assumptions)


def test_plans_explicit_year_frequency():
    plan = plan_query(
        "Show average fare_amount by year over pickup_datetime",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )

    assert plan.operation == "time_series"
    assert plan.frequency == "1y"


@pytest.mark.parametrize("year_word", ["yearly", "annual", "annually"])
def test_plans_explicit_yearly_frequency_without_daily_fallback(year_word):
    plan = plan_query(
        f"Show {year_word} average fare_amount over pickup_datetime",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )

    assert plan.operation == "time_series"
    assert plan.frequency == "1y"
    assert plan.agg == "mean"
    assert not any("daily" in assumption.lower() for assumption in plan.assumptions)


def test_plans_hourly_average_without_trend_keyword():
    plan = plan_query(
        "Show hourly average fare_amount over pickup_datetime",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )

    assert plan.operation == "time_series"
    assert plan.frequency == "1h"
    assert plan.metric == "fare_amount"
    assert plan.agg == "mean"


def test_plans_hourly_trend_with_average():
    plan = plan_query(
        "Show hourly trend of average fare_amount over pickup_datetime",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )

    assert plan.operation == "time_series"
    assert plan.frequency == "1h"
    assert plan.metric == "fare_amount"
    assert plan.agg == "mean"


def test_grouped_count_plan_matches_analysis_api():
    plan = plan_query(
        "Count trips by payment_type",
        columns=COLUMNS,
        numeric_columns=[],
        datetime_columns=[],
    )
    df = pl.DataFrame({"payment_type": [1, 1, 2]})

    assert plan.operation == "aggregate"
    assert plan.agg == "count"
    assert plan.to_dict()["agg"] == "count"
    assert plan.to_dict()["aggregation"] == "count"
    assert plan.aggregation == "count"
    result = aggregate(df, group_by=plan.group_by, metric=plan.metric, agg=plan.agg)
    assert result["result"] == [
        {"payment_type": 1, "row_count": 2},
        {"payment_type": 2, "row_count": 1},
    ]


def test_time_series_plan_parameters_match_analysis_api():
    plan = plan_query(
        "Show the daily average fare_amount over pickup_datetime",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    df = pl.DataFrame(
        {
            "pickup_datetime": [
                "2025-01-01T10:00:00",
                "2025-01-01T12:00:00",
            ],
            "fare_amount": [10.0, 20.0],
        }
    ).with_columns(pl.col("pickup_datetime").str.to_datetime())

    result = time_series(
        df,
        datetime_column=plan.datetime_column,
        frequency=plan.frequency,
        metric=plan.metric,
        agg=plan.agg,
    )

    assert result["result"][0]["fare_amount"] == 15.0


def test_cli_plan_emits_json_without_loading_dataset_rows(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "sample.csv"
    dataset.write_text("payment_type,fare_amount\n1,10.0\n2,20.0\n", encoding="utf-8")
    monkeypatch.setattr(
        "data_analyst.cli.load_dataset",
        lambda _: pytest.fail("planning should inspect schema without loading all rows"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["data-analyst", "plan", str(dataset), "average fare_amount by payment_type"],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["operation"] == "aggregate"
    assert result["group_by"] == ["payment_type"]
    assert result["metric"] == "fare_amount"
    assert result["agg"] == "mean"


def test_rejects_ambiguous_correlation():
    try:
        plan_query("Are these variables correlated?", columns=COLUMNS, numeric_columns=NUMERIC)
    except PlanningError as exc:
        assert "two explicitly named numeric columns" in str(exc)
    else:
        raise AssertionError("Expected PlanningError")


def test_rejects_unknown_request():
    try:
        plan_query("Predict the future revenue", columns=COLUMNS, numeric_columns=NUMERIC)
    except PlanningError as exc:
        assert "Unsupported analytical request" in str(exc)
    else:
        raise AssertionError("Expected PlanningError")


def test_does_not_invent_columns():
    plan = plan_query(
        "What is the average fare_amount by payment_type?",
        columns=COLUMNS,
        numeric_columns=NUMERIC,
        datetime_columns=DATETIME,
    )
    assert all(value in COLUMNS for value in plan.group_by + ([plan.metric] if plan.metric else []))


def test_plans_grouped_query_with_overlapping_metric_names():
    plan = plan_query(
        "What is the average GDP growth by country?",
        columns=["country", "country_code", "year", "gdp_growth", "gdp", "population"],
        numeric_columns=["year", "gdp_growth", "gdp", "population"],
        datetime_columns=[],
    )
    assert plan.operation == "aggregate"
    assert plan.group_by == ["country"]
    assert plan.metric == "gdp_growth"
    assert plan.aggregation == "mean"





def test_plans_ranking_query_with_overlapping_metric_names():
    plan = plan_query(
        "Which country has the highest average GDP growth?",
        columns=["country", "country_code", "year", "gdp_growth", "gdp", "population"],
        numeric_columns=["year", "gdp_growth", "gdp", "population"],
        datetime_columns=[],
    )
    assert plan.operation == "aggregate"
    assert plan.group_by == ["country"]
    assert plan.metric == "gdp_growth"
    assert plan.aggregation == "mean"
    assert plan.sort_direction == "desc"




def test_plans_filtered_year_time_series():
    plan = plan_query(
        "How did India's GDP growth change over the years?",
        columns=["country", "country_code", "year", "gdp_growth", "gdp", "population"],
        numeric_columns=["year", "gdp_growth", "gdp", "population"],
        datetime_columns=[],
    )
    assert plan.operation == "time_series"
    assert plan.datetime_column == "year"
    assert plan.metric == "gdp_growth"
    assert plan.aggregation == "mean"
    assert plan.frequency == "1y"
    assert plan.filter_column == "country"
    assert plan.filter_value == "India"



def test_plans_filtered_year_time_series_with_explicit_average():
    plan = plan_query(
        "What was India's average GDP growth over the years?",
        columns=["country", "country_code", "year", "gdp_growth", "gdp", "population"],
        numeric_columns=["year", "gdp_growth", "gdp", "population"],
        datetime_columns=[],
    )
    assert plan.operation == "time_series"
    assert plan.datetime_column == "year"
    assert plan.metric == "gdp_growth"
    assert plan.aggregation == "mean"
    assert plan.frequency == "1y"
    assert plan.filter_column == "country"
    assert plan.filter_value == "India"




def test_plans_filtered_year_time_series_with_explicit_sum():
    plan = plan_query(
        "Show the sum of India's GDP growth over the years",
        columns=["country", "year", "gdp_growth"],
        numeric_columns=["year", "gdp_growth"],
        datetime_columns=[],
    )
    assert plan.operation == "time_series"
    assert plan.datetime_column == "year"
    assert plan.metric == "gdp_growth"
    assert plan.aggregation == "sum"
    assert plan.frequency == "1y"
    assert plan.filter_column == "country"
    assert plan.filter_value == "India"
