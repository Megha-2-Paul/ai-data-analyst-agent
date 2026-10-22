"""Public portfolio demo endpoints are isolated from authenticated storage."""
from fastapi.testclient import TestClient

from data_analyst.api.main import create_app


def demo_client(tmp_path):
    return TestClient(create_app(
        database_url=f"sqlite:///{tmp_path / 'demo.sqlite'}",
        storage_dir=tmp_path / "uploads",
        secret_key="demo-test-secret-long-enough-for-tests",
        auto_create_tables=True,
    ))


def test_demo_dataset_and_profile_are_public_and_use_synthetic_data(tmp_path):
    with demo_client(tmp_path) as client:
        sample = client.get("/api/v1/demo/dataset")
        assert sample.status_code == 200
        assert sample.json()["is_sample"] is True
        assert sample.json()["rows"] == 120

        profile = client.post("/api/v1/demo/profile")
        assert profile.status_code == 200
        body = profile.json()
        assert body["source"] == "generated_synthetic_sample"
        assert body["dataset"]["rows"] == 120
        assert body["profile"]["columns"]
        assert "completeness_pct" in body["quality"]


def test_demo_analysis_executes_python_engine_without_login(tmp_path):
    with demo_client(tmp_path) as client:
        response = client.post(
            "/api/v1/demo/analyses",
            data={"question": "Which payment_type has the highest average fare_amount?"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source"] == "generated_synthetic_sample"
        assert body["response"]["answer"]["evidence"]
        assert body["response"]["execution_steps"]


def test_demo_upload_is_parsed_in_memory_and_not_persisted(tmp_path):
    with demo_client(tmp_path) as client:
        response = client.post(
            "/api/v1/demo/profile",
            files={"file": ("small.csv", b"category,revenue\na,10\nb,20\na,30\n", "text/csv")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source"] == "uploaded_file_in_memory"
        assert body["dataset"]["rows"] == 3
        assert body["dataset"]["column_names"] == ["category", "revenue"]
        assert list((tmp_path / "uploads").iterdir()) == []


def test_demo_upload_limits_and_supported_formats(tmp_path):
    with demo_client(tmp_path) as client:
        unsupported = client.post(
            "/api/v1/demo/profile",
            files={"file": ("sample.xlsx", b"not an excel file", "application/octet-stream")},
        )
        assert unsupported.status_code == 422
        oversized = client.post(
            "/api/v1/demo/profile",
            files={"file": ("large.csv", b"x" * (5 * 1024 * 1024 + 1), "text/csv")},
        )
        assert oversized.status_code == 413


def test_demo_profile_can_use_prepared_dataset_without_mutating_upload(tmp_path):
    with demo_client(tmp_path) as client:
        response = client.post(
            "/api/v1/demo/profile",
            data={"use_prepared": "true"},
            files={"file": ("sample.csv", b"Customer Name,order_date,amount\n Alice ,2025-01-01,10\nBob,2025-01-02,20\n", "text/csv")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["dataset"]["column_names"] == ["customer_name", "order_date", "amount"]
        assert body["dataset"]["rows"] == 2
        assert body["source"] == "uploaded_file_in_memory"


def test_demo_narrative_endpoint_returns_grounded_fallback_without_groq(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with demo_client(tmp_path) as client:
        response = client.post("/api/v1/demo/narrative", json={
            "report": {
                "report": {
                    "original_profile": {"row_count": 4, "column_count": 2},
                    "prepared_profile": {"row_count": 4, "column_count": 2},
                    "original_quality": {"null_counts": {"amount": 1}, "duplicate_row_count": 0},
                    "cleaning": {"applied_steps": [{"operation": "trim_strings"}], "recommendations": ["Review missing amount values."]},
                }
            }
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["provider"] == "local_fallback"
        assert "4 before and 4 after" in body["narrative"]
        assert "Review missing amount values." in body["narrative"]


def test_demo_narrative_rejects_invalid_or_oversized_reports(tmp_path):
    with demo_client(tmp_path) as client:
        invalid = client.post("/api/v1/demo/narrative", json={"report": "not a report object"})
        assert invalid.status_code == 422
        oversized = client.post("/api/v1/demo/narrative", json={"report": {"blob": "x" * 100_001}})
        assert oversized.status_code == 413
