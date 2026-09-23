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
