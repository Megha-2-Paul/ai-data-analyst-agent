"""API integration tests using an isolated SQLite database and temporary storage."""

from fastapi.testclient import TestClient

from data_analyst.api.database import Base, make_engine
from data_analyst.api.main import create_app


def client_for(tmp_path):
    db_path = tmp_path / "test.sqlite"
    engine = make_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    app = create_app(
        database_url=f"sqlite:///{db_path}",
        storage_dir=tmp_path / "uploads",
        secret_key="test-secret-key-that-is-long-enough-for-testing",
    )
    return TestClient(app)


def register(client, email):
    response = client.post("/api/v1/auth/register", json={"email": email, "password": "secure-pass-123"})
    assert response.status_code == 201, response.text
    token = client.post("/api/v1/auth/login", json={"email": email, "password": "secure-pass-123"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload_csv(client, headers, name="sales.csv"):
    return client.post(
        "/api/v1/datasets", headers=headers,
        files={"file": (name, b"category,revenue\na,10\nb,20\na,30\n", "text/csv")},
    )


def test_auth_dataset_analysis_report_and_usage_flow(tmp_path):
    with client_for(tmp_path) as client:
        headers = register(client, "analyst@example.com")
        upload = upload_csv(client, headers)
        assert upload.status_code == 201, upload.text
        dataset_id = upload.json()["id"]

        profile = client.get(f"/api/v1/datasets/{dataset_id}/profile", headers=headers)
        assert profile.status_code == 200
        assert profile.json()["row_count"] == 3

        analysis = client.post(
            f"/api/v1/datasets/{dataset_id}/analyses", headers=headers,
            json={"question": "What is the average revenue by category?", "planner": "deterministic"},
        )
        assert analysis.status_code == 201, analysis.text
        analysis_id = analysis.json()["id"]
        assert analysis.json()["response"]["answer"]["evidence"]

        report = client.post(f"/api/v1/analyses/{analysis_id}/report", headers=headers)
        assert report.status_code == 201, report.text
        report_id = report.json()["id"]
        download = client.get(f"/api/v1/reports/{report_id}/download", headers=headers)
        assert download.status_code == 200
        assert "application/json" in download.headers["content-type"]

        usage = client.get("/api/v1/usage", headers=headers)
        assert usage.status_code == 200
        assert any(row["action"] == "analysis" for row in usage.json()["events"])


def test_requires_authentication_and_rejects_invalid_token(tmp_path):
    with client_for(tmp_path) as client:
        assert client.get("/api/v1/datasets").status_code == 401
        assert client.get("/api/v1/datasets", headers={"Authorization": "Bearer bad"}).status_code == 401


def test_user_cannot_access_another_users_dataset(tmp_path):
    with client_for(tmp_path) as client:
        owner_headers = register(client, "owner@example.com")
        other_headers = register(client, "other@example.com")
        dataset = upload_csv(client, owner_headers).json()
        dataset_id = dataset["id"]
        assert client.get(f"/api/v1/datasets/{dataset_id}", headers=other_headers).status_code == 404
        assert client.get(f"/api/v1/datasets/{dataset_id}/profile", headers=other_headers).status_code == 404
        assert client.delete(f"/api/v1/datasets/{dataset_id}", headers=other_headers).status_code == 404


def test_upload_validation_and_size_limit(tmp_path, monkeypatch):
    with client_for(tmp_path) as client:
        headers = register(client, "uploads@example.com")
        bad = client.post("/api/v1/datasets", headers=headers, files={"file": ("bad.txt", b"hello", "text/plain")})
        assert bad.status_code == 422
        monkeypatch.setenv("MAX_UPLOAD_BYTES", "10")
        too_large = client.post(
            "/api/v1/datasets", headers=headers,
            files={"file": ("large.csv", b"category,revenue\na,123456789", "text/csv")},
        )
        assert too_large.status_code == 413


def test_dataset_preparation_does_not_change_uploaded_source(tmp_path):
    with client_for(tmp_path) as client:
        headers = register(client, "prep@example.com")
        dataset = upload_csv(client, headers).json()
        before = client.get(f"/api/v1/datasets/{dataset['id']}/profile", headers=headers).json()
        prepared = client.post(f"/api/v1/datasets/{dataset['id']}/prepare", headers=headers)
        after = client.get(f"/api/v1/datasets/{dataset['id']}/profile", headers=headers).json()
        assert prepared.status_code == 200
        assert before == after


def test_provider_without_key_returns_controlled_service_error(tmp_path, monkeypatch):
    with client_for(tmp_path) as client:
        headers = register(client, "provider@example.com")
        dataset = upload_csv(client, headers).json()
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        response = client.post(
            f"/api/v1/datasets/{dataset['id']}/analyses", headers=headers,
            json={"question": "What is the average revenue by category?", "planner": "groq"},
        )
        assert response.status_code == 503
