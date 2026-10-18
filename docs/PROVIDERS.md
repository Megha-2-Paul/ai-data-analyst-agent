# Optional AI providers and free-tier smoke tests

Groq and Gemini are optional planners. The deterministic planner remains the default, so standard tests do not require external credentials.

## Configure

Copy `.env.example` to `.env` and set keys from the providers' official consoles. Never commit a real key or paste it into source files.

- Groq: `GROQ_API_KEY`; model override `GROQ_MODEL` (default `openai/gpt-oss-120b`).
- Gemini: `GEMINI_API_KEY`; model override `GEMINI_MODEL` (default `gemini-2.5-flash`).

API requests select a provider with `{"planner":"groq"}` or `{"planner":"gemini"}`. CLI supports `--planner groq` and `--planner gemini`.

## Test layers

- `pytest tests/test_providers.py` runs offline tests with fake responses and consumes no provider quota.
- `pytest tests/test_live_providers.py` sends a small real request only for each configured key. Missing-key tests are skipped.
- `pytest` runs the whole suite; live tests skip unless credentials are present.

For live tests in GitHub Actions, add `GROQ_API_KEY` and/or `GEMINI_API_KEY` under repository Settings → Secrets and variables → Actions. Then run the manual **Optional provider smoke tests** workflow. Optional model overrides can be set as Actions variables. Keys must never be placed in workflow YAML.

Free tiers have model-specific and account-specific quotas that can change. A successful smoke test verifies the configured key/model at that time; it does not guarantee future availability or unlimited free use. Provider errors are reduced to safe error categories rather than logging credentials or full request contents. The provider's own data-handling terms apply to schema/question text sent to its API; do not send sensitive data without reviewing those terms.
