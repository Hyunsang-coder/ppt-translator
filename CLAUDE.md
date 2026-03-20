# CLAUDE.md

## Project Overview

PPT 번역캣 — PowerPoint translation app (LangChain + OpenAI/Anthropic). FastAPI backend (`api.py`) + Next.js frontend (`frontend/`). Preserves original formatting, glossary support, auto language detection, real-time progress via polling.

**Python >= 3.12** required.

## Development Commands

```bash
# Backend
uvicorn api:app --reload --port 8000
pytest tests/ -v                    # all tests
pytest tests/ -v -m slow            # slow tests (API calls)

# Frontend
cd frontend && npm install
cd frontend && npm run dev
cd frontend && npm run build
cd frontend && npx tsc --noEmit     # type check
```

## Claude Code Commands

- `/commit` — Review & commit
- `/push` — Push with safety checks
- `/dev-backend` / `/dev-frontend` — Start dev servers
- `/deploy-ec2` — Deploy backend to EC2
- `/check-ec2` — Check EC2 status
- `/update-docs` — Update CLAUDE.md
- `/update-patch-notes` — Update patch notes from git

### Agents
- `code-reviewer` — Code quality & security review
- `test-runner` — Run tests concisely
- `translation-qa` — Translation logic & prompt review

## Reference Docs

Detailed documentation in `docs/`:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Backend/frontend structure, components, tests
- [`docs/KEY_PATTERNS.md`](docs/KEY_PATTERNS.md) — Translation flow, job flow, text fit, thread safety, error handling
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — Env vars, deployment, models, libraries, API usage
