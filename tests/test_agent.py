import polars as pl

from data_analyst.agent import AnalystAgent


def make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "payment_type": ["card", "cash", "card", "cash"],
            "fare_amount": [10.0, 20.0, 30.0, 40.0],
            "trip_distance": [1.0, 2.0, 3.0, 4.0],
            "pickup_datetime": pl.datetime_range(
                pl.datetime(2025, 1, 1),
                pl.datetime(2025, 1, 4),
                interval="1d",
                eager=True,
            ),
        }
    )


def test_agent_executes_grouped_question():
    response = AnalystAgent().ask(
        "What is the average fare_amount by payment_type?",
        make_df(),
    )
    assert response.plan.operation == "aggregate"
    assert response.plan.metric == "fare_amount"
    assert response.result["analysis"] == "grouped_aggregation"
    assert len(response.result["result"]) == 2
    assert response.execution_steps[-1] == "return_structured_result"


def test_agent_executes_time_series_question():
    response = AnalystAgent().ask(
        "Show the daily trend of average fare_amount over pickup_datetime",
        make_df(),
    )
    assert response.plan.operation == "time_series"
    assert response.result["analysis"] == "time_series"


def test_agent_executes_correlation():
    response = AnalystAgent().ask(
        "What is the correlation between fare_amount and trip_distance?",
        make_df(),
    )
    assert response.plan.operation == "correlation"
    assert response.result["analysis"] == "correlation"


def test_agent_executes_top_n():
    response = AnalystAgent().ask(
        "Show the top 1 payment_type groups by highest average fare_amount",
        make_df(),
    )
    assert response.plan.limit == 1
    assert len(response.result["result"]) == 1


def test_agent_serializes_trace():
    response = AnalystAgent().ask(
        "Give me descriptive statistics for trip_distance",
        make_df(),
    )
    payload = response.to_dict()
    assert payload["question"]
    assert payload["plan"]["operation"] == "describe"
    assert payload["execution_steps"] == [
        "inspect_dataset_schema",
        "plan:describe",
        "validate_plan",
        "execute_analysis",
        "return_structured_result",
    ]


def test_agent_executes_filtered_year_time_series():
    df = pl.DataFrame({
        "country": ["India", "India", "USA", "India"],
        "year": [2020, 2021, 2020, 2022],
        "gdp_growth": [3.0, 4.0, 2.0, 5.0],
    })
    response = AnalystAgent().ask(
        "How did India's GDP growth change over the years?",
        df,
    )
    assert response.plan.operation == "time_series"
    assert response.plan.datetime_column == "year"
    assert response.plan.metric == "gdp_growth"
    assert response.plan.filter_column == "country"
    assert response.plan.filter_value == "India"
    assert response.result["analysis"] == "time_series"
    assert response.result["result"] == [
        {"year": 2020, "gdp_growth": 3.0},
        {"year": 2021, "gdp_growth": 4.0},
        {"year": 2022, "gdp_growth": 5.0},
    ]
