"""FastAPI application exposing the existing analytical engine securely."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import jwt
import polars as pl
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from data_analyst.agent import AnalystAgent
from data_analyst.ingestion import load_dataset
from data_analyst.llm_planner import LLMPlanningError, OpenAIPlanner
from data_analyst.planner import PlanningError, plan_query
from data_analyst.preparation import prepare_dataset
from data_analyst.profiling import profile_dataset
from data_analyst.quality import quality_report
from data_analyst.providers import GeminiPlanner, GroqPlanner

from .database import Base, make_engine, make_session_factory
from .models import AnalysisRun, Dataset, Report, UsageEvent, User
from .schemas import AnalysisRequest, DatasetResponse, LoginRequest, RegisterRequest, ReportResponse, TokenResponse, UserResponse
from .security import create_access_token, decode_access_token, hash_password, verify_password
from .storage import resolve_storage_key, save_and_inspect, storage_root

bearer = HTTPBearer(auto_error=False)
API_PREFIX = "/api/v1"


def create_app(
    *,
    database_url: str | None = None,
    storage_dir: str | Path | None = None,
    secret_key: str | None = None,
    auto_create_tables: bool = False,
) -> FastAPI:
    environment = os.getenv("APP_ENV", "development").lower()
    configured_secret = secret_key or os.getenv("APP_SECRET_KEY")
    if environment == "production" and (not configured_secret or len(configured_secret) < 32):
        raise RuntimeError("APP_SECRET_KEY must be configured with at least 32 characters in production.")
    configured_secret = configured_secret or "local-development-secret-change-before-deployment"
    db_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./ai_data_analyst.db")
    root = storage_root(storage_dir)
    engine = make_engine(db_url)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.auto_create_tables:
            Base.metadata.create_all(bind=application.state.engine)
        try:
            yield
        finally:
            application.state.engine.dispose()

    app = FastAPI(
        lifespan=lifespan,
        title="AI Data Analyst Agent API",
        version="0.2.0",
        description="Authenticated dataset analysis with deterministic calculations and optional AI planning.",
    )
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.storage_dir = root
    app.state.secret_key = configured_secret
    app.state.auto_create_tables = auto_create_tables

    def get_db() -> Generator[Session, None, None]:
        db = app.state.session_factory()
        try:
            yield db
        finally:
            db.close()

    def current_user(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
        db: Session = Depends(get_db),
    ) -> User:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Authentication required.", headers={"WWW-Authenticate": "Bearer"})
        try:
            user_id = decode_access_token(credentials.credentials, request.app.state.secret_key)
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired access token.", headers={"WWW-Authenticate": "Bearer"})
        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid or expired access token.", headers={"WWW-Authenticate": "Bearer"})
        return user

    def owned_dataset(db: Session, user: User, dataset_id: int) -> Dataset:
        dataset = db.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.owner_id == user.id))
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return dataset

    def dataset_frame(dataset: Dataset) -> pl.DataFrame:
        try:
            path = resolve_storage_key(dataset.storage_key, app.state.storage_dir)
        except ValueError:
            raise HTTPException(status_code=404, detail="Dataset file is unavailable.")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Dataset file is unavailable.")
        try:
            return load_dataset(path)
        except (OSError, ValueError):
            raise HTTPException(status_code=422, detail="The stored dataset could not be read.")

    def record_usage(db: Session, user_id: int, action: str, resource_type: str, resource_id: int | None, units: int = 1):
        db.add(UsageEvent(user_id=user_id, action=action, resource_type=resource_type, resource_id=resource_id, units=units))

    def configured_planner(name: str):
        if name == "deterministic":
            return plan_query
        env_key = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}[name]
        if not os.getenv(env_key):
            raise HTTPException(status_code=503, detail=f"The {name} planner is selected but {env_key} is not configured.")
        if name == "groq":
            return GroqPlanner()
        if name == "gemini":
            return GeminiPlanner()
        return OpenAIPlanner()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def readiness(db: Session = Depends(get_db)):
        try:
            db.execute(text("SELECT 1"))
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail="Database is not ready.")
        return {"status": "ready"}

    @app.post(f"{API_PREFIX}/auth/register", response_model=UserResponse, status_code=201)
    def register(payload: RegisterRequest, db: Session = Depends(get_db)):
        if db.scalar(select(User.id).where(User.email == payload.email)):
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        user = User(email=payload.email, password_hash=hash_password(payload.password))
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except IntegrityError:
            db.rollback()
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        return user

    @app.post(f"{API_PREFIX}/auth/login", response_model=TokenResponse)
    def login(payload: LoginRequest, db: Session = Depends(get_db)):
        user = db.scalar(select(User).where(User.email == payload.email))
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        return TokenResponse(access_token=create_access_token(user.id, app.state.secret_key))

    @app.get(f"{API_PREFIX}/auth/me", response_model=UserResponse)
    def me(user: User = Depends(current_user)):
        return user

    @app.post(f"{API_PREFIX}/datasets", response_model=DatasetResponse, status_code=201)
    async def upload_dataset(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        user: User = Depends(current_user),
    ):
        original_name = Path(file.filename or "dataset").name[:255]
        try:
            suffix = Path(original_name).suffix.lower()
            if suffix not in {".csv", ".parquet"}:
                raise HTTPException(status_code=422, detail="Only CSV and Parquet files are supported.")
            max_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = await file.read(min(1024 * 1024, max_bytes + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"Upload exceeds the {max_bytes}-byte limit.")
                chunks.append(chunk)
            content = b"".join(chunks)
            try:
                key, df, _ = save_and_inspect(content, original_name, app.state.storage_dir)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            dataset = Dataset(
                owner_id=user.id, original_filename=original_name, storage_key=key,
                format=suffix[1:], size_bytes=total, row_count=df.height, columns_json=df.columns,
            )
            db.add(dataset)
            db.flush()
            record_usage(db, user.id, "dataset_upload", "dataset", dataset.id, total)
            db.commit()
            db.refresh(dataset)
            return DatasetResponse(
                id=dataset.id, original_filename=dataset.original_filename, format=dataset.format,
                size_bytes=dataset.size_bytes, row_count=dataset.row_count,
                columns=dataset.columns_json, created_at=dataset.created_at,
            )
        finally:
            await file.close()

    @app.get(f"{API_PREFIX}/datasets", response_model=list[DatasetResponse])
    def list_datasets(
        limit: int = 50, offset: int = 0,
        db: Session = Depends(get_db), user: User = Depends(current_user),
    ):
        if limit < 1 or limit > 100 or offset < 0:
            raise HTTPException(status_code=422, detail="limit must be 1–100 and offset must be non-negative.")
        rows = db.scalars(
            select(Dataset).where(Dataset.owner_id == user.id).order_by(Dataset.id.desc()).offset(offset).limit(limit)
        ).all()
        return [DatasetResponse(
            id=d.id, original_filename=d.original_filename, format=d.format,
            size_bytes=d.size_bytes, row_count=d.row_count, columns=d.columns_json, created_at=d.created_at,
        ) for d in rows]

    @app.get(f"{API_PREFIX}/datasets/{{dataset_id}}", response_model=DatasetResponse)
    def get_dataset(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        d = owned_dataset(db, user, dataset_id)
        return DatasetResponse(
            id=d.id, original_filename=d.original_filename, format=d.format,
            size_bytes=d.size_bytes, row_count=d.row_count, columns=d.columns_json, created_at=d.created_at,
        )

    @app.delete(f"{API_PREFIX}/datasets/{{dataset_id}}", status_code=204)
    def delete_dataset(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        d = owned_dataset(db, user, dataset_id)
        path = resolve_storage_key(d.storage_key, app.state.storage_dir)
        path.unlink(missing_ok=True)
        db.delete(d)
        record_usage(db, user.id, "dataset_delete", "dataset", dataset_id)
        db.commit()
        return Response(status_code=204)

    @app.get(f"{API_PREFIX}/datasets/{{dataset_id}}/profile")
    def profile(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        return profile_dataset(dataset_frame(owned_dataset(db, user, dataset_id)))

    @app.get(f"{API_PREFIX}/datasets/{{dataset_id}}/quality")
    def quality(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        return quality_report(dataset_frame(owned_dataset(db, user, dataset_id)))

    @app.post(f"{API_PREFIX}/datasets/{{dataset_id}}/prepare")
    def prepare(dataset_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        d = owned_dataset(db, user, dataset_id)
        result = prepare_dataset(dataset_frame(d))
        record_usage(db, user.id, "dataset_prepare", "dataset", dataset_id)
        db.commit()
        return result.to_dict()

    @app.post(f"{API_PREFIX}/datasets/{{dataset_id}}/analyses", status_code=201)
    def create_analysis(
        dataset_id: int, payload: AnalysisRequest,
        db: Session = Depends(get_db), user: User = Depends(current_user),
    ):
        d = owned_dataset(db, user, dataset_id)
        max_daily = max(1, int(os.getenv("MAX_DAILY_ANALYSES", "100")))
        since = datetime.now(timezone.utc) - timedelta(days=1)
        count = db.scalar(select(func.count(UsageEvent.id)).where(
            UsageEvent.user_id == user.id, UsageEvent.action == "analysis",
            UsageEvent.created_at >= since,
        )) or 0
        if count >= max_daily:
            raise HTTPException(status_code=429, detail="Daily analysis limit reached.")
        df = dataset_frame(d)
        prep_payload = None
        if payload.prepare:
            prep = prepare_dataset(df)
            df = prep.dataframe
            prep_payload = prep.to_dict()
        planner = configured_planner(payload.planner)
        try:
            response = AnalystAgent(planner=planner).ask(payload.question, df)
            result = response.to_dict()
        except LLMPlanningError as exc:
            if "planning request failed" in str(exc).lower():
                raise HTTPException(status_code=503, detail="The selected AI provider is unavailable or rate limited.")
            raise HTTPException(status_code=422, detail=str(exc))
        except PlanningError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception:
            raise HTTPException(status_code=422, detail="The analysis request could not be completed for this dataset.")
        if prep_payload is not None:
            result["preparation"] = prep_payload
        run = AnalysisRun(
            owner_id=user.id, dataset_id=d.id, question=payload.question,
            planner=payload.planner, status="succeeded", response_json=result,
        )
        db.add(run)
        db.flush()
        record_usage(db, user.id, "analysis", "analysis", run.id)
        db.commit()
        db.refresh(run)
        return {"id": run.id, "dataset_id": d.id, "status": run.status, "created_at": run.created_at, "response": result}

    @app.get(f"{API_PREFIX}/analyses")
    def list_analyses(
        limit: int = 50, offset: int = 0,
        db: Session = Depends(get_db), user: User = Depends(current_user),
    ):
        if limit < 1 or limit > 100 or offset < 0:
            raise HTTPException(status_code=422, detail="limit must be 1–100 and offset must be non-negative.")
        rows = db.scalars(
            select(AnalysisRun).where(AnalysisRun.owner_id == user.id)
            .order_by(AnalysisRun.id.desc()).offset(offset).limit(limit)
        ).all()
        return [{"id": r.id, "dataset_id": r.dataset_id, "question": r.question, "planner": r.planner,
                 "status": r.status, "created_at": r.created_at} for r in rows]

    @app.get(f"{API_PREFIX}/analyses/{{analysis_id}}")
    def get_analysis(analysis_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        run = db.scalar(select(AnalysisRun).where(AnalysisRun.id == analysis_id, AnalysisRun.owner_id == user.id))
        if run is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return {"id": run.id, "dataset_id": run.dataset_id, "question": run.question, "planner": run.planner,
                "status": run.status, "created_at": run.created_at, "response": run.response_json}

    @app.post(f"{API_PREFIX}/analyses/{{analysis_id}}/report", response_model=ReportResponse, status_code=201)
    def create_report(analysis_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        run = db.scalar(select(AnalysisRun).where(AnalysisRun.id == analysis_id, AnalysisRun.owner_id == user.id))
        if run is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        payload = {
            "report_version": 1, "analysis_id": run.id, "dataset_id": run.dataset_id,
            "question": run.question, "planner": run.planner, "created_at": run.created_at.isoformat(),
            "response": run.response_json,
        }
        report = Report(
            owner_id=user.id, dataset_id=run.dataset_id, analysis_id=run.id,
            title=f"Analysis report #{run.id}", payload_json=payload,
        )
        db.add(report)
        db.flush()
        record_usage(db, user.id, "report_create", "report", report.id)
        db.commit()
        db.refresh(report)
        return report

    @app.get(f"{API_PREFIX}/reports", response_model=list[ReportResponse])
    def list_reports(
        limit: int = 50, offset: int = 0,
        db: Session = Depends(get_db), user: User = Depends(current_user),
    ):
        if limit < 1 or limit > 100 or offset < 0:
            raise HTTPException(status_code=422, detail="limit must be 1–100 and offset must be non-negative.")
        return db.scalars(
            select(Report).where(Report.owner_id == user.id)
            .order_by(Report.id.desc()).offset(offset).limit(limit)
        ).all()

    @app.get(f"{API_PREFIX}/reports/{{report_id}}", response_model=ReportResponse)
    def get_report(report_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        report = db.scalar(select(Report).where(Report.id == report_id, Report.owner_id == user.id))
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found.")
        return report

    @app.get(f"{API_PREFIX}/reports/{{report_id}}/download")
    def download_report(report_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
        report = db.scalar(select(Report).where(Report.id == report_id, Report.owner_id == user.id))
        if report is None:
            raise HTTPException(status_code=404, detail="Report not found.")
        body = json.dumps(report.payload_json, ensure_ascii=False, indent=2, allow_nan=False)
        return Response(
            content=body, media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="analysis-report-{report.id}.json"'},
        )

    @app.get(f"{API_PREFIX}/usage")
    def usage(db: Session = Depends(get_db), user: User = Depends(current_user)):
        rows = db.execute(
            select(UsageEvent.action, func.count(UsageEvent.id), func.coalesce(func.sum(UsageEvent.units), 0))
            .where(UsageEvent.user_id == user.id).group_by(UsageEvent.action)
        ).all()
        return {"user_id": user.id, "events": [
            {"action": action, "event_count": count, "units": units}
            for action, count, units in rows
        ]}

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(request: Request, exc: SQLAlchemyError):
        return Response(
            content='{"detail":"A database operation failed."}',
            status_code=503, media_type="application/json",
        )

    return app


app = create_app()
