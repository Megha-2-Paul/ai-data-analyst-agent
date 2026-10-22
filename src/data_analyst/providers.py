"""Optional Groq and Gemini query planners with shared strict plan validation."""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from .llm_planner import LLMPlanningError, _schema_context, _validate_plan_payload
from .planner import QueryPlan

_SYSTEM = (
    "You are a conservative data-analysis query planner. Return only a JSON object. "
    "Never invent column names. Never return SQL, Python, code, or executable instructions. "
    "Choose only supported operations and fields."
)


def _prompt(question: str, columns, numeric_columns, datetime_columns) -> str:
    return (
        f"Question: {question}\n"
        f"Schema: {_schema_context(columns, numeric_columns, datetime_columns)}\n"
        "Allowed operations: profile, quality, describe, aggregate, correlation, time_series.\n"
        "Allowed aggregations: count, sum, mean, median, min, max, std.\n"
        "Allowed frequencies: 1h, 1d, 1w, 1mo, 1y.\n"
        "Return JSON fields: operation, columns, group_by, metric, aggregation, datetime_column, "
        "frequency, sort_direction, limit, assumptions, filter_column, filter_value. "
        "Use null for absent scalar fields and empty arrays for absent array fields."
    )


class GroqPlanner:
    """Optional Groq planner using the official Groq Python client."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model or (os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not os.getenv("GROQ_API_KEY"):
            raise LLMPlanningError("GROQ_API_KEY is not configured.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMPlanningError("Groq SDK is missing; reinstall the project dependencies.") from exc
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=30.0, max_retries=1)
        return self._client

    def __call__(
        self, question: str, *, columns: Sequence[str],
        numeric_columns: Sequence[str] | None = None,
        datetime_columns: Sequence[str] | None = None,
    ) -> QueryPlan:
        try:
            response = self._get_client().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _prompt(question, columns, numeric_columns, datetime_columns)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            payload = json.loads(response.choices[0].message.content or "")
        except LLMPlanningError:
            raise
        except json.JSONDecodeError as exc:
            raise LLMPlanningError("Groq returned invalid JSON.") from exc
        except Exception as exc:
            raise LLMPlanningError(f"Groq planning request failed ({type(exc).__name__}).") from exc
        return _validate_plan_payload(
            payload, question=question, columns=columns,
            numeric_columns=numeric_columns, datetime_columns=datetime_columns,
        )


class GeminiPlanner:
    """Optional Gemini planner using the official Google Gen AI Python SDK."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model or (os.getenv("GEMINI_MODEL") or "gemini-3.8-flash")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not os.getenv("GEMINI_API_KEY"):
            raise LLMPlanningError("GEMINI_API_KEY is not configured.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMPlanningError("Gemini SDK is missing; reinstall the project dependencies.") from exc
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=30_000))
        return self._client

    def __call__(
        self, question: str, *, columns: Sequence[str],
        numeric_columns: Sequence[str] | None = None,
        datetime_columns: Sequence[str] | None = None,
    ) -> QueryPlan:
        try:
            response = self._get_client().models.generate_content(
                model=self._model,
                contents=_prompt(question, columns, numeric_columns, datetime_columns),
                config={
                    "system_instruction": _SYSTEM,
                    "response_mime_type": "application/json",
                    "temperature": 0,
                },
            )
            payload = json.loads(response.text or "")
        except LLMPlanningError:
            raise
        except json.JSONDecodeError as exc:
            raise LLMPlanningError("Gemini returned invalid JSON.") from exc
        except Exception as exc:
            raise LLMPlanningError(f"Gemini planning request failed ({type(exc).__name__}).") from exc
        return _validate_plan_payload(
            payload, question=question, columns=columns,
            numeric_columns=numeric_columns, datetime_columns=datetime_columns,
        )
