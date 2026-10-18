# Backend API

The FastAPI backend wraps the existing analytical engine. It never executes model-generated code. The deterministic planner is the default and needs no external API key.

## Local setup and secure configuration

1. Use Python 3.11 or newer and create a virtual environment.
2. Run `pip install -e ".[dev]"`.
3. Copy `.env.example` to `.env`; set a unique `APP_SECRET_KEY` and a database URL.
4. Configure PostgreSQL, or use SQLite for a local-only trial.
5. Run `alembic upgrade head`.
6. Start with `uvicorn data_analyst.api.main:app --env-file .env --reload`.
7. Open `http://127.0.0.1:8000/docs` for interactive OpenAPI docs.

For Docker, copy `.env.example` to `.env`, replace `APP_SECRET_KEY` and `POSTGRES_PASSWORD` with unique values, then run `docker compose up --build`. Use a URL-safe database password because Compose embeds it in `DATABASE_URL`. The Compose service runs in production mode, rejects the example secret, and runs migrations before starting the API.

## Authentication

Register with `POST /api/v1/auth/register`, then log in with `POST /api/v1/auth/login`. Send the returned token in `Authorization: Bearer <token>`. Tokens expire after 60 minutes. Passwords are hashed and not stored in plaintext.

## Routes

- `GET /health` — process liveness; `GET /ready` — database readiness.
- `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `GET /api/v1/auth/me`.
- `GET /api/v1/workspaces`, `POST /api/v1/workspaces`, `GET /api/v1/workspaces/{id}/members`, `POST /api/v1/workspaces/{id}/members`.
- `POST /api/v1/datasets` — multipart CSV/Parquet upload.
- `GET /api/v1/datasets`, `GET /api/v1/datasets/{id}`, `DELETE /api/v1/datasets/{id}`.
- `GET /api/v1/datasets/{id}/profile`, `GET /api/v1/datasets/{id}/quality`, `POST /api/v1/datasets/{id}/prepare`.
- `POST /api/v1/datasets/{id}/analyses` — body: `{"question":"...","planner":"deterministic","prepare":false}`.
- `GET /api/v1/analyses`, `GET /api/v1/analyses/{id}`.
- `POST /api/v1/analyses/{id}/report`, `GET /api/v1/reports`, `GET /api/v1/reports/{id}`, `GET /api/v1/reports/{id}/download`.
- `GET /api/v1/usage`.

The planner field accepts `deterministic`, `groq`, `gemini`, or `openai`. Provider use requires the matching API key. Reports are downloadable JSON containing the question, result, findings, evidence and limitations.

## Limits and security

The default upload limit is 25 MiB (`MAX_UPLOAD_BYTES`) and the default analysis limit is 100 runs per user per rolling 24 hours (`MAX_DAILY_ANALYSES`). New users receive a personal workspace. Workspace owners can add already-registered users; members can collaborate on datasets, analyses and reports within that workspace. Only the uploader or workspace owner can delete a dataset. Dataset files use random storage keys, are not served as static files, and are only opened after workspace membership checks. API errors do not disclose filesystem paths or database details.

Before public deployment, configure HTTPS, a unique application secret and database password, backups, restricted network access, storage quotas/retention, monitoring, request-rate limiting at a reverse proxy/API gateway, and a privacy policy. The application enforces a per-user analysis quota, but that is not a substitute for edge-level rate limiting on authentication and upload endpoints. This initial API processes analysis synchronously and uses local file storage; very large or long-running workloads may need background workers and object storage.
