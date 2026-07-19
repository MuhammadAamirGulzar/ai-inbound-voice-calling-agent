# Contributing

## Scope

This repository is maintained with a product-first standard: reliability, readability, and operational safety.

## Development Workflow

1. Create a feature branch from `main`.
2. Keep changes focused and small.
3. Run relevant checks before opening a pull request.
4. Open a PR with a clear summary, test notes, and rollback considerations.

## Commit Convention

Use Conventional Commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation-only changes
- `refactor:` code restructuring without behavior change
- `chore:` maintenance tasks

Examples:
- `feat: add websocket fallback handling for interrupted audio streams`
- `fix: hash user passwords before persistence`
- `docs: document deployment and runtime configuration`

## Pull Request Checklist

- Behavior is tested locally for affected paths.
- No secrets or generated artifacts are introduced.
- Documentation is updated for user-facing changes.
- Database or API contract changes are clearly documented.
