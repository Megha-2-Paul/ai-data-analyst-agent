# Development Workflow

This document is the source of truth for collaboration, repository changes, review, and Git safety. Re-check live GitHub state before every consequential operation; branch and pull-request status can change.

## Roles and collaboration

- **Codex / coding agent** inspects the repository, implements approved changes, runs tests, and performs Git operations within the approved scope.
- **ChatGPT** helps audit the repository, review plans and diffs, interpret test results, and assess risks.
- **The user** owns product and engineering decisions and approves implementation plans and consequential actions, including commits, pushes, merges, pull-request changes, and branch cleanup. A specific authorization (such as a direct commit to `main`) applies only to the stated scope.

## Plan-first approval

1. **Audit first.** Inspect the relevant files, tests, history, branch state, open pull requests, and CI results before proposing changes.
2. **Propose a concrete plan.** State the goal, scope, files, implementation approach, risks, dependencies, and acceptance criteria.
3. **Wait for approval.** Do not implement before the user approves the plan and delivery method.
4. **Stay in scope.** Ask for approval before adding unrelated work, changing architecture, or moving into a later stage.
5. **Verify and report.** Review the full diff, run appropriate checks, and report what changed, what was tested, limitations, and the resulting commit or PR.

## Git and branch safety

- Check the current branch, worktree status, remote state, and target branch before editing.
- Prefer a focused task branch for implementation and review; avoid creating redundant branches.
- A user may explicitly authorize a narrowly scoped direct commit to `main`. When they do, keep the change within that scope, verify the target branch and current file version, and do not bundle unrelated work.
- Preserve uncommitted user changes. Never discard, overwrite, reset, or stash them without explicit authorization.
- Never force-push, reset, rebase, merge, delete branches, close or retarget pull requests, or rewrite history without explicit authorization.
- Keep commits focused and descriptive. Commit only the files and changes covered by the approval.
- Fetching remote references and changing local branch state are separate operations. Fetching updates remote-tracking references; it does not itself switch or integrate a local branch.
- Before updating an existing file through GitHub, fetch its latest content and blob SHA. If the target has changed since the audit, re-read it and reconcile rather than overwriting newer work.

## Testing and review

- For code changes, run relevant targeted tests, then the full test suite when appropriate.
- Run `git diff --check` and review the complete final diff when working in a local Git checkout.
- Investigate failures. Never hide, skip, or weaken tests simply to obtain a passing result.
- Report exact test commands and results. Do not claim tests passed unless they were actually run and passed.
- Check the applicable GitHub Actions / CI results before recommending a merge.
- For documentation-only changes, verify links, stage descriptions, PR status, and consistency with the live repository. State clearly when application tests were not run because no code changed.

## Pull-request and completion workflow

- Review the entire PR diff, its base and head branches, related dependencies, and checks before recommending integration.
- A PR being closed or merged into a feature branch does **not** mean its work is included in `main`. Verify the target branch and ancestry.
- Merge only when explicitly authorized.
- After an authorized merge, fetch remote references and safely synchronize local `main`; use fast-forward-only integration when possible.
- Verify commit ancestry, branch divergence, and worktree cleanliness after synchronization.
- Close superseded PRs or delete obsolete branches only with explicit authorization, and confirm current GitHub state first.
- Finish with a concise report of files changed, validation performed, limitations, and the commit SHA / PR URL.

## Stage boundaries and architecture

- Complete, validate, and review each stage before beginning the next.
- Keep documentation-only changes separate from feature implementation.
- The deterministic Python analytical engine remains responsible for data validation, calculations, and analytical results. Any LLM may assist with natural-language interpretation or planning, but must not silently replace the engine as the source of truth.
- AI integrations must be optional, with credentials stored outside source control, graceful handling of provider errors and rate limits, and a deterministic path when the provider is unavailable.
- Keep existing CLI and Python interfaces stable unless an approved plan explicitly changes them.
- Treat workflow automation (including a possible n8n integration) as a separately scoped design decision. First define the use case and boundaries; preserve the existing Python engine and interfaces, and introduce an API boundary such as FastAPI only if the approved architecture and requirements justify it. Do not add n8n, FastAPI, or related dependencies as part of unrelated work.

## Secrets and external services

- Keep API keys and credentials in `.env` or an appropriate secret store.
- Never commit real secrets. Use `.env.example` only for placeholder variable names and safe example values.
- Avoid exposing secrets in logs, test output, screenshots, issues, or commits.
- External-service integrations must handle unavailable services, malformed responses, authentication failures, timeouts, rate limits, and other API errors. Deterministic operation should remain possible when the service is optional.

## Current project checkpoint

This checkpoint reflects the repository audit recorded on 2026-10-18. Re-check live GitHub state before acting on any PR or branch.

### `main` baseline

- Stage 1.3 query planning and validation is merged into `main` through PR #10.
- The Stage 1.3 fix commit was `3a704c4a4f77b645eaaea92e7d458668f67783a6`; the PR #10 merge commit was `0e322ace92e28e6dc945a5c5081a2bf8a27602ac`.
- The README on `main` describes Stage 1.4 optional Groq integration as planned, not implemented on `main`.
- The repository CI workflow runs the test suite on pushes and pull requests using Python 3.11 and pytest. Confirm the latest run rather than relying on historical results.

### Pull requests and feature branches

- PR #4 (Stage 1.3 query planning) is closed and merged. PR #10 contains the later Stage 1.3 hardening fix and is also closed and merged. Do not rely on stale notes that describe PR #4 as still open.
- PR #5, **Stage 1.4 agent orchestration**, was reported open with base `stage-1.3-query-planning` and head `stage-1.4-agent-orchestration`.
- PR #6 (LLM planner), PR #7 (multi-step analytical reasoning), and PR #8 (evidence-backed answer generation) were reported merged into feature branches, not into `main`. Their presence in feature history must not be treated as proof that those stages are integrated into `main`.
- PR #9, **Stage 4 data preparation**, was reported open with base `stage-3-evidence-answer` and head `stage-4-data-preparation`.
- Feature branches have substantial divergence from `main`. Before integration, compare branch ancestry and diffs, inspect overlapping changes and tests, and agree on an integration sequence. Do not merge or clean up branches as part of a status audit.
- The approved backend scope on stage-4-data-preparation includes optional Groq and Gemini planner adapters alongside the existing OpenAI adapter. Keep provider credentials external to source control, preserve deterministic planning, and run live provider smoke tests only when a key is configured.

### Next-stage guardrails

1. Audit current `main`, feature branches, open PRs, and CI before selecting the next implementation task.
2. Decide which feature-branch work should be reviewed and integrated before beginning additional functionality.
3. For changes to the approved Groq, Gemini, or OpenAI planner adapters, preserve the shared QueryPlan validation contract, deterministic fallback, bounded requests, safe error handling, and offline plus opt-in live tests.
4. For a possible n8n workflow, define the user-facing workflow and data flow first. Keep calculations and validation in the existing Python engine; expose or wrap interfaces only through a separately approved design.
5. Do not assume that a planned feature is implemented, that a feature-branch merge updates `main`, or that an old CI result validates newer commits.


## Backend implementation checkpoint — 2026-10-18

The user approved the consolidated Phases 2B–2E backend plan for implementation on the existing stage-4-data-preparation branch. The current implementation includes a FastAPI boundary, PostgreSQL-compatible persistence and initial Alembic migration, token authentication, per-user resource ownership checks, CSV/Parquet upload, analysis history, JSON reports, usage events, quotas, and optional Groq/Gemini planner adapters. The implementation remains on the feature branch; main must remain unchanged and PR #9 must not be merged without separate authorization.

Standard CI uses fake provider responses. Live Groq/Gemini smoke tests require repository Actions secrets and should not be reported as completed when keys are absent. Finish by validating migrations, API isolation, provider failure handling, Docker setup, documentation, and the full CI result before the user pulls the branch.
