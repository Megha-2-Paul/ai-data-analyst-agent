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


class DatasetResponse(BaseModel):
    id: int
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
    dataset_id: int
    analysis_id: int
    title: str
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
