# Development Workflow

This document records the agreed collaboration, review, and Git safety practices for the repository.

## Roles and collaboration

- **Codex** inspects the repository, implements approved changes, runs tests, and performs Git operations when authorized.
- **ChatGPT** helps review plans, code changes, test reports, risks, and decisions.
- **The user** approves implementation plans and consequential actions, including commits, pushes, merges, pull-request changes, and branch cleanup.

## Plan-first approval

1. Audit the current repository state and relevant code, tests, documentation, and history before editing.
2. Present the intended scope, files, approach, risks, and acceptance criteria.
3. Wait for user approval before implementation.
4. Keep implementation within the approved scope. Ask again before expanding it.

## Git and branch safety

- Check the current branch and working-tree status before starting.
- Use an appropriate task branch for implementation work; avoid unnecessary duplicate branches.
- Preserve uncommitted user changes. Never discard, overwrite, reset, or stash them without explicit authorization.
- Do not force-push, reset, rebase, merge, delete branches, or modify pull requests without explicit authorization.
- Keep commits focused and descriptive. Commit only approved files and changes.
- Treat fetching remote references and changing local branch state as separate operations. Fetching updates remote-tracking references; it does not itself switch or integrate a local branch.

## Testing and code review

- Run relevant targeted tests, then the full test suite when appropriate.
- Run `git diff --check` and review the complete final diff.
- Investigate failures. Do not hide, skip, or weaken tests to obtain a passing result.
- Report exact test commands and results. Do not claim a test passed unless it was run and passed.
- Check the applicable GitHub Actions/CI results before recommending a merge.

## Pull-request and completion workflow

- Summarize changed files, behavior, test commands/results, limitations, and commit SHA.
- Review pull-request changes and checks before merging.
- Merge only when explicitly authorized.
- After a merge, fetch remote references and safely synchronize local `main`; use fast-forward-only integration when possible.
- Verify commit ancestry, branch divergence, and worktree cleanliness after synchronization.
- Close superseded pull requests and delete obsolete branches only with explicit authorization; confirm current GitHub state first.
- Report the final repository state and wait for the next task.

## Stage boundaries

- Complete, validate, and review each stage before beginning the next.
- Keep documentation-only changes separate from feature implementation.
- Do not silently expand the approved scope or introduce future-stage code early.

## Secrets and external services

- Keep API keys and credentials in `.env` or an appropriate secret store.
- Never commit real secrets.
- Use `.env.example` only for placeholder variable names and safe example values.
- Avoid exposing secrets in logs, test output, screenshots, or commits.
- Integrations with external services must account for unavailable services, malformed responses, rate limits, and other API errors; deterministic operation should remain possible when the service is optional.

## Current project checkpoint

- Stage 1.3 query planning and validation is merged into GitHub `main` by PR #10.
- Stage 1.3 fix commit: `3a704c4a4f77b645eaaea92e7d458668f67783a6`.
- PR #10 merge commit: `0e322ace92e28e6dc945a5c5081a2bf8a27602ac`.
- Local `main` was synchronized to `origin/main` at that merge commit and verified clean in the audit for this documentation update. Recheck live repository state before future operations.
- PR #4 has been reported as closed and merged in earlier project notes, but the live GitHub API check for this update reports it open and unmerged, with head `stage-1.3-query-planning` and base `main`. Preserve the history and do not modify or close it without explicit authorization.
- Stage 1.4 optional Groq integration is planned, not implemented. It requires a separate approved plan, implementation, and test cycle.
