# PROJECT_MEMORY

### Future Autonomous Grok Factory (tabling — April 2026)
Vision preserved for when we have a purpose-built machine (Mac mini / Mac Studio with more RAM):
- Grok-style supervisor (Gemma 4 via MLX) that plans, delegates, and reviews
- Full handoff to local Codex (Aider) and Antigravity (CLI/MCP)
- Zero user input after high-level goal
- Sequential execution to stay within memory limits
- Full reuse of existing teamAI supervisor loop, evals, deterministic patches, learned notes, MockBackend, handoff.py, etc.
- All ideas from the recent conversation thread are retained here for future implementation.

## Purpose
- Personal local-first project for a closed-loop AI system on Apple Silicon.
- Goal: turn a small local model into a useful supervised coding coprocessor that can reduce reliance on higher-cost remote reasoning for coding tasks.

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
  - deterministic narrow-write compiler for explicit small edits
  - approval-gated write path through `.teamai/approvals/`
  - persistent workspace memory and learned-note ranking/pruning
  - bridge-based local execution plus Codex handoff flow
  - FastAPI endpoints, CLI entrypoint, and streaming event output

## Repo State
- Active implementation now lives in `teamai/`
- Packaged via `pyproject.toml`
- `.env.example` provides local runtime settings
- tests use stdlib `unittest`
- public GitHub repo exists
- Apache-2.0 licensed

## Recent Accomplishments
- Improved robustness of JSON planning / verification (`json_utils.py`)
- Added persistent run history and workspace memory (`memory.py`)
- Added learned-note ranking, task-aware ranking, and self-pruning memory behavior
- Added safer patch-oriented write tools and approval gating (`tools.py`, `approvals.py`)
- Added approval continuation flow with scoped verification
- Added streaming events for CLI, API, and jobs (`events.py`, `api.py`, `jobs.py`)
- Added bridge launch/status flow for real Terminal-side local runs (`bridge.py`)
- Added memory-aware bridge profiles and automatic retry on memory pressure
- Added route-aware handoff generation for broader tasks (`handoff.py`)
- Added a real eval harness with isolated subprocess execution (`evals.py`)
- Added runtime-health-aware eval scoring so infra failures are not misread as agent failures
- Fixed eval subprocess interpreter selection on macOS so live local evals use the real venv runtime
- Hardened the deterministic append smoke path so the narrow write smoke case now reaches `approval_required`

## Current Maturity
- The project is now a real supervised local coding coprocessor, not just scaffolding.
- Broad autonomous coding is still intentionally limited; broader tasks should route to reconnaissance plus handoff.
- Latest live smoke eval on the real local runtime is meaningful:
  - runtime health: `healthy (mlx_import_ok)`
  - pass rate: `2/3`
  - actionable pass rate: `2/3`
  - passing cases:
    - deterministic narrow patch approval flow
    - broad Codex handoff flow
  - remaining failing case:
    - repository inspection timing out under the inspection profile

## Current Build Targets
- Reduce inspection timeout and drift for repo-inspection tasks.
- Improve inspection/recon retrieval, ranking, and early-stop behavior.
- Further harden structured output reliability and convergence on local planning turns.
- Continue expanding deterministic compiler coverage for explicit narrow coding chores.
- Use eval feedback to keep improving local-model routing and usefulness over time.

## Known Constraints
- 16GB memory means model choice must stay modest.
- Larger local models may be too slow or memory-heavy.
- MLX imports still crash in the current Codex desktop environment, so in-process local inference from this environment is not reliable.
- Real unsandboxed local evals on the Mac now run and produce meaningful results; the main blocker is no longer runtime availability.
- Closed-loop autonomy needs safety controls before enabling broad writes or shell execution.

## Important Convention
- Keep the project local-first.
- Writes stay disabled by default.
- `workspace_write` mode only works when config also enables writes.
- Narrow explicit write tasks should prefer the deterministic compiler path and stop at `approval_required`.
- Broad implementation tasks should prefer safe reconnaissance plus Codex handoff over pretending the small local model can carry them end to end.
- Avoid reintroducing cloud-specific abstractions unless they are explicitly optional.
