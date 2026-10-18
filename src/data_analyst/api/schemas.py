"""Validated API request and response schemas."""

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("A valid email address is required.")
        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    @field_validator("name")
    @classmethod
    def workspace_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Workspace name cannot be blank.")
        return value.strip()


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    role: str
    created_at: datetime


class WorkspaceMemberCreate(BaseModel):
    email: str = Field(min_length=5, max_length=320)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class WorkspaceMemberResponse(BaseModel):
    email: str
    role: str


class DatasetResponse(BaseModel):
    id: int
    workspace_id: int
    original_filename: str
    format: str
    size_bytes: int
    row_count: int
    columns: list[str]
    created_at: datetime


class AnalysisRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    planner: str = "deterministic"
    prepare: bool = False

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value.strip()

    @field_validator("planner")
    @classmethod
    def planner_supported(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"deterministic", "groq", "gemini", "openai"}:
            raise ValueError("planner must be deterministic, groq, gemini, or openai.")
        return value


class ReportResponse(BaseModel):
    id: int
    workspace_id: int
    dataset_id: int
    analysis_id: int
    title: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MLModelRequest(BaseModel):
    target: str = Field(min_length=1, max_length=200)
    features: list[str] = Field(min_length=1, max_length=100)
    task: str = "regression"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)

    @field_validator("target")
    @classmethod
    def normalize_target(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target cannot be blank.")
        return value

    @field_validator("features")
    @classmethod
    def normalize_features(cls, values: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not normalized:
            raise ValueError("At least one non-blank feature is required.")
        return normalized

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification.")
        return value



class ForecastRequest(BaseModel):
    datetime_column: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    horizon: int = Field(default=1, ge=1, le=24)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    n_lags: int = Field(default=3, ge=1, le=12)

    @field_validator("datetime_column", "target")
    @classmethod
    def normalize_column_names(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Column names cannot be blank.")
        return value
