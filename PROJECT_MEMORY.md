# PROJECT_MEMORY

## Purpose
- Personal local-first project for a closed-loop AI system on Apple Silicon.
- Goal: local LLM personas strategize and plan while executable agents inspect, edit, test, and verify work locally.

## Current Scope
- Private/local usage only
- Apple Silicon target machine
- MLX-first inference stack
- No required cloud APIs
- Local FastAPI service plus CLI

## Hardware Target
- M1 Pro MacBook Pro
- 16GB unified memory

## Direction
- Prefer local models over hosted APIs to minimize ongoing token spend.
- Current default model target: `mlx-community/gemma-4-e4b-it-4bit`
- Current architecture:
  - MLX model backend via `mlx-vlm`
  - local strategist / critic / coder-planner / verifier loop
  - local tool agents for files, search, reads, shell, and optional writes
  - FastAPI endpoints and CLI entrypoint

## Repo State
- Active implementation now lives in `teamai/`
- Packaged via `pyproject.toml`
- `.env.example` provides local runtime settings
- tests use stdlib `unittest`

## Next Build Target
- Improve robustness of JSON planning / verification
- Add persistent run history
- Add safer patch-oriented write tools
- Add approval checkpoints before destructive changes
- Add streaming events

## Known Constraints
- 16GB memory means model choice must stay modest.
- Larger local models may be too slow or memory-heavy.
- MLX imports crash in the current Codex desktop environment, so local inference could not be executed here even though the code is wired for it.
- Closed-loop autonomy needs safety controls before enabling broad writes or shell execution.

## Important Convention
- Keep the project local-first.
- Writes stay disabled by default.
- `workspace_write` mode only works when config also enables writes.
- Avoid reintroducing cloud-specific abstractions unless they are explicitly optional.
