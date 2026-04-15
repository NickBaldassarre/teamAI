from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..approvals import PatchApprovalStore
from ..codex_prompts import build_codex_handoff_prompt
from ..memory import WorkspaceMemoryStore
from ..sandbox import Sandbox
from ..schemas import CodexHandoffPayload, HandoffArtifact, HandoffRevisionRequest, RateLimitState
from ..verification import VerificationResult, verify_patch

DEFAULT_FAILURE_CONTEXT_FILE = ".teamai/failure_context.log"
_HANDOFF_HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)(pyproject\.toml|package\.json|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|poetry\.lock|requirements(\.txt)?|setup\.py|Pipfile(\.lock)?)$", re.IGNORECASE),
    re.compile(r"(^|/)(migrations?|alembic|schema|schemas?)(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)\.env(\..+)?$", re.IGNORECASE),
    re.compile(r"(^|/)(\.github|\.gitlab|\.circleci|vercel\.json|netlify\.toml|Dockerfile|docker-compose)", re.IGNORECASE),
)


@dataclass(frozen=True)
class BridgeModelResponse:
    patch_text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    rate_limit_state: RateLimitState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgeExecutionResult:
    engine: str
    model: str
    payload_file: Path
    patch_file: Path
    prompt: str
    patch_text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    rate_limit_state: RateLimitState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifiedBridgeExecutionResult:
    execution: BridgeExecutionResult
    verification: VerificationResult
    failure_context_file: Path
    approval: dict[str, Any] | None = None
    approval_error: str | None = None
    artifact: HandoffArtifact | None = None
    revision_requests: tuple[HandoffRevisionRequest, ...] = ()
    accepted: bool = False
    revision_count: int = 0


class AgentBridge(ABC):
    engine: str = "unknown"
    default_model: str = ""
    approval_source_tool: str = "verified_codex_handoff"

    def execute(
        self,
        *,
        project_root: Path,
        payload_file: str | Path,
        patch_file: str | Path,
        model: str | None = None,
    ) -> BridgeExecutionResult:
        project_root = project_root.resolve()
        payload_path = _resolve_project_path(project_root, payload_file)
        patch_path = _resolve_project_path(project_root, patch_file)

        payload = CodexHandoffPayload.model_validate_json(payload_path.read_text(encoding="utf-8"))
        prompt = build_codex_handoff_prompt(payload)
        model_name = (model or self.default_model or "").strip()
        if not model_name:
            raise RuntimeError(f"No model is configured for bridge engine `{self.engine}`.")

        response = _coerce_bridge_response(
            self._request_patch(
                prompt=prompt,
                model=model_name,
                payload=payload,
                payload_path=payload_path,
                project_root=project_root,
            )
        )
        patch_text = _sanitize_patch_output(response.patch_text)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(_ensure_trailing_newline(patch_text), encoding="utf-8")
        if response.rate_limit_state is not None:
            try:
                WorkspaceMemoryStore().update_rate_limits(model_id=model_name, state=response.rate_limit_state)
            except Exception:
                pass

        return BridgeExecutionResult(
            engine=self.engine,
            model=model_name,
            payload_file=payload_path,
            patch_file=patch_path,
            prompt=prompt,
            patch_text=patch_text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            rate_limit_state=response.rate_limit_state,
            metadata=dict(response.metadata),
        )

    def execute_verified(
        self,
        *,
        project_root: Path,
        payload_file: str | Path,
        patch_file: str | Path,
        failure_context_file: str | Path = DEFAULT_FAILURE_CONTEXT_FILE,
        model: str | None = None,
        max_revision_attempts: int = 0,
        create_approval: bool = True,
    ) -> VerifiedBridgeExecutionResult:
        project_root = project_root.resolve()
        payload_path = _resolve_project_path(project_root, payload_file)
        patch_path = _resolve_project_path(project_root, patch_file)
        failure_context_path = _resolve_project_path(project_root, failure_context_file)
        approval: dict[str, Any] | None = None
        approval_error: str | None = None
        payload = CodexHandoffPayload.model_validate_json(payload_path.read_text(encoding="utf-8"))
        base_prompt = build_codex_handoff_prompt(payload)
        model_name = (model or self.default_model or "").strip()
        if not model_name:
            raise RuntimeError(f"No model is configured for bridge engine `{self.engine}`.")

        revision_requests: list[HandoffRevisionRequest] = []
        accepted = False
        artifact: HandoffArtifact | None = None
        execution: BridgeExecutionResult | None = None
        verification: VerificationResult | None = None

        for attempt in range(max_revision_attempts + 1):
            prompt = _build_revision_prompt(base_prompt=base_prompt, revision_request=revision_requests[-1] if revision_requests else None)
            response = _coerce_bridge_response(
                self._request_patch(
                prompt=prompt,
                model=model_name,
                payload=payload,
                payload_path=payload_path,
                project_root=project_root,
                )
            )
            if response.rate_limit_state is not None:
                try:
                    WorkspaceMemoryStore().update_rate_limits(model_id=model_name, state=response.rate_limit_state)
                except Exception:
                    pass
            artifact = _parse_handoff_artifact(response.patch_text, engine=self.engine)
            patch_text = _ensure_trailing_newline(_sanitize_patch_output(artifact.diff))
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(patch_text, encoding="utf-8")
            execution = BridgeExecutionResult(
                engine=self.engine,
                model=model_name,
                payload_file=payload_path,
                patch_file=patch_path,
                prompt=prompt,
                patch_text=patch_text,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
                rate_limit_state=response.rate_limit_state,
                metadata=dict(response.metadata),
            )

            with Sandbox(project_root) as sandbox:
                verification = verify_patch(execution.patch_file, sandbox)
                revision_request = _build_revision_request(
                    artifact=artifact,
                    payload=payload,
                    verification=verification,
                )
                if revision_request is None:
                    accepted = True
                    if create_approval:
                        try:
                            approval = PatchApprovalStore().create_bundle_from_patch(
                                workspace=project_root,
                                sandbox_root=sandbox.path,
                                patch_text=execution.patch_text,
                                reason=f"Verified {self.engine} handoff patch for: {payload.original_task}",
                                source_tool=self.approval_source_tool,
                                continuation={
                                    "original_task": payload.original_task,
                                    "requested_execution_mode": "workspace_write",
                                    "source_engine": self.engine,
                                },
                            )
                        except Exception as exc:  # pragma: no cover - defensive surface
                            approval_error = str(exc)
                    break

            revision_requests.append(revision_request)
            if attempt >= max_revision_attempts:
                break

        if execution is None or verification is None:
            raise RuntimeError("Verified handoff did not produce an execution attempt.")

        if accepted:
            if failure_context_path.exists():
                failure_context_path.unlink()
        else:
            failure_context_path.parent.mkdir(parents=True, exist_ok=True)
            failure_context_path.write_text(
                _ensure_trailing_newline(
                    _render_revision_failure_context(
                        verification=verification,
                        revision_request=revision_requests[-1] if revision_requests else None,
                    )
                ),
                encoding="utf-8",
            )

        return VerifiedBridgeExecutionResult(
            execution=execution,
            verification=verification,
            failure_context_file=failure_context_path,
            approval=approval,
            approval_error=approval_error,
            artifact=artifact,
            revision_requests=tuple(revision_requests),
            accepted=accepted,
            revision_count=len(revision_requests),
        )

    @abstractmethod
    def _request_patch(
        self,
        *,
        prompt: str,
        model: str,
        payload: CodexHandoffPayload,
        payload_path: Path,
        project_root: Path,
    ) -> BridgeModelResponse | str:
        raise NotImplementedError


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def _coerce_bridge_response(response: BridgeModelResponse | str) -> BridgeModelResponse:
    if isinstance(response, BridgeModelResponse):
        return response
    return BridgeModelResponse(patch_text=str(response))


def build_rate_limit_state(
    *,
    provider: str,
    model_id: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    headers: dict[str, Any] | None = None,
    source: str = "usage_only",
) -> RateLimitState:
    normalized_headers = {str(key).lower(): value for key, value in (headers or {}).items()}
    effective_total = total_tokens if total_tokens is not None else _sum_defined(prompt_tokens, completion_tokens)
    remaining_requests = _first_header_int(
        normalized_headers,
        "x-ratelimit-remaining-requests",
        "anthropic-ratelimit-requests-remaining",
        "ratelimit-remaining-requests",
    )
    requests_limit = _first_header_int(
        normalized_headers,
        "x-ratelimit-limit-requests",
        "anthropic-ratelimit-requests-limit",
        "ratelimit-limit-requests",
    )
    remaining_tokens = _first_header_int(
        normalized_headers,
        "x-ratelimit-remaining-tokens",
        "anthropic-ratelimit-tokens-remaining",
        "ratelimit-remaining-tokens",
    )
    tokens_limit = _first_header_int(
        normalized_headers,
        "x-ratelimit-limit-tokens",
        "anthropic-ratelimit-tokens-limit",
        "ratelimit-limit-tokens",
    )
    remaining_daily_requests = _first_header_int(
        normalized_headers,
        "x-ratelimit-remaining-requests-day",
        "x-ratelimit-remaining-day-requests",
    )
    daily_requests_limit = _first_header_int(
        normalized_headers,
        "x-ratelimit-limit-requests-day",
        "x-ratelimit-limit-day-requests",
    )
    remaining_daily_tokens = _first_header_int(
        normalized_headers,
        "x-ratelimit-remaining-tokens-day",
        "x-ratelimit-remaining-day-tokens",
    )
    daily_tokens_limit = _first_header_int(
        normalized_headers,
        "x-ratelimit-limit-tokens-day",
        "x-ratelimit-limit-day-tokens",
    )

    state = RateLimitState(
        provider=provider,
        model_id=model_id,
        requests_made=1,
        tokens_in=max(prompt_tokens or 0, 0),
        tokens_out=max(completion_tokens or max((effective_total or 0) - max(prompt_tokens or 0, 0), 0), 0),
        requests_limit=requests_limit,
        remaining_requests=remaining_requests,
        tokens_limit=tokens_limit,
        remaining_tokens=remaining_tokens,
        daily_requests_limit=daily_requests_limit,
        remaining_daily_requests=remaining_daily_requests,
        daily_tokens_limit=daily_tokens_limit,
        remaining_daily_tokens=remaining_daily_tokens,
        source=source,
    )

    headrooms = []
    if requests_limit and remaining_requests is not None:
        headrooms.append(max(0.0, min(1.0, remaining_requests / requests_limit)))
    if tokens_limit and remaining_tokens is not None:
        headrooms.append(max(0.0, min(1.0, remaining_tokens / tokens_limit)))
    state.window_headroom = min(headrooms) if headrooms else None

    daily_headrooms = []
    if daily_requests_limit and remaining_daily_requests is not None:
        daily_headrooms.append(max(0.0, min(1.0, remaining_daily_requests / daily_requests_limit)))
    if daily_tokens_limit and remaining_daily_tokens is not None:
        daily_headrooms.append(max(0.0, min(1.0, remaining_daily_tokens / daily_tokens_limit)))
    state.daily_headroom = min(daily_headrooms) if daily_headrooms else None

    all_headrooms = [value for value in (state.window_headroom, state.daily_headroom) if value is not None]
    state.quota_pressure = round(1.0 - min(all_headrooms), 4) if all_headrooms else 0.0
    return state


def extract_headers(response: Any) -> dict[str, Any]:
    direct_headers = getattr(response, "headers", None)
    if isinstance(direct_headers, dict):
        return direct_headers
    if hasattr(direct_headers, "items"):
        return {str(key): value for key, value in direct_headers.items()}

    raw_response = getattr(response, "_response", None)
    raw_headers = getattr(raw_response, "headers", None)
    if isinstance(raw_headers, dict):
        return raw_headers
    if hasattr(raw_headers, "items"):
        return {str(key): value for key, value in raw_headers.items()}
    return {}


def extract_usage_counts(
    usage: Any,
    *,
    prompt_keys: tuple[str, ...] = ("input_tokens", "prompt_tokens"),
    completion_keys: tuple[str, ...] = ("output_tokens", "completion_tokens"),
    total_keys: tuple[str, ...] = ("total_tokens",),
) -> tuple[int | None, int | None, int | None]:
    prompt_tokens = _first_attr_or_key_int(usage, prompt_keys)
    completion_tokens = _first_attr_or_key_int(usage, completion_keys)
    total_tokens = _first_attr_or_key_int(usage, total_keys)
    if total_tokens is None:
        total_tokens = _sum_defined(prompt_tokens, completion_tokens)
    return prompt_tokens, completion_tokens, total_tokens


def _parse_handoff_artifact(text: str, *, engine: str) -> HandoffArtifact:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "diff" in payload:
        artifact_payload = dict(payload)
        artifact_payload.setdefault("engine", engine)
        return HandoffArtifact.model_validate(artifact_payload)
    return HandoffArtifact(
        engine=engine,
        summary=f"{engine} handoff returned a raw diff patch.",
        diff=_sanitize_patch_output(stripped),
        rationale="The bridge engine returned only a patch, so no structured rationale was provided.",
        confidence=0.6,
    )


def _sanitize_patch_output(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    if not any(marker in stripped for marker in ("diff --git", "--- ", "+++ ")):
        raise RuntimeError("Bridge response did not look like a unified diff patch.")
    return stripped


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else f"{text}\n"


def _first_header_int(headers: dict[str, Any], *names: str) -> int | None:
    for name in names:
        if name not in headers:
            continue
        value = _coerce_int(headers.get(name))
        if value is not None:
            return value
    return None


def _first_attr_or_key_int(source: Any, names: tuple[str, ...]) -> int | None:
    if source is None:
        return None
    for name in names:
        value = getattr(source, name, None)
        if value is None and isinstance(source, dict):
            value = source.get(name)
        coerced = _coerce_int(value)
        if coerced is not None:
            return coerced
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _sum_defined(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return max(left or 0, 0) + max(right or 0, 0)


def _build_revision_prompt(*, base_prompt: str, revision_request: HandoffRevisionRequest | None) -> str:
    if revision_request is None:
        return base_prompt
    reasons = ", ".join(revision_request.reasons) or "unspecified issue"
    details = revision_request.details.strip() or "No extra detail supplied."
    return (
        f"{base_prompt}\n\n"
        "Revision request:\n"
        f"- Reasons: {reasons}\n"
        f"- Details: {details}\n"
        "Return a corrected handoff artifact that stays within scope and fixes the validation issue."
    )


def _build_revision_request(
    *,
    artifact: HandoffArtifact,
    payload: CodexHandoffPayload,
    verification: VerificationResult,
) -> HandoffRevisionRequest | None:
    reasons: list[str] = []
    details: list[str] = []
    touched_paths = _extract_patch_paths(artifact.diff)

    if not verification.success:
        reasons.append("failing_tests")
        details.append(_format_revision_detail("Failing test excerpts", _extract_failing_test_excerpt(verification.log_output)))
    if artifact.confidence < 0.35:
        reasons.append("verifier_objections")
        details.append(
            _format_revision_detail(
                "Verifier concerns",
                f"Handoff confidence {artifact.confidence:.2f} is below the acceptance threshold.",
            )
        )
    if any(_path_is_high_risk(path) for path in touched_paths):
        reasons.append("policy_violations")
        details.append(
            _format_revision_detail(
                "Policy violations",
                f"Handoff touched high-risk paths: {', '.join(touched_paths)}",
            )
        )
    if payload.core_dependencies and touched_paths and not _artifact_is_raw_diff_fallback(artifact):
        if all(not _path_matches_scope(path, payload.core_dependencies) for path in touched_paths):
            reasons.extend(["incorrect_scope", "path_scope_violations"])
            details.append(
                _format_revision_detail(
                    "Path or scope violations",
                    "Touched paths do not intersect the scoped dependencies: "
                    + ", ".join(payload.core_dependencies),
                )
            )

    deduped_reasons = list(dict.fromkeys(reasons))
    if not deduped_reasons:
        return None
    return HandoffRevisionRequest(
        reasons=deduped_reasons,
        details=(
            "\n\n".join(part for part in details if part).strip()
            or _format_revision_detail("Rejected patch", "The patch failed verification for an unspecified reason.")
        ),
    )


def _render_revision_failure_context(
    *,
    verification: VerificationResult,
    revision_request: HandoffRevisionRequest | None,
) -> str:
    if revision_request is None:
        return verification.log_output
    reasons = list(revision_request.reasons)
    detail = revision_request.details.strip() or verification.log_output.strip() or "Handoff validation failed."
    if reasons == ["failing_tests"] and detail == (verification.log_output.strip() or detail):
        return verification.log_output
    reason_text = ", ".join(reasons) or "unspecified"
    return f"Revision required ({reason_text})\n{detail}"


def _extract_failing_test_excerpt(log_output: str) -> str:
    lines = [line.rstrip() for line in log_output.splitlines() if line.strip()]
    if not lines:
        return "Sandbox verification failed without any captured output."
    selected: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in ("fail", "error", "traceback", "assert", "expected", "exit_code")):
            selected.append(line)
    excerpt = selected[-10:] if selected else lines[-10:]
    return "\n".join(excerpt)


def _format_revision_detail(title: str, body: str) -> str:
    normalized = body.strip() or "No detail available."
    return f"{title}:\n{normalized}"


def _extract_patch_paths(diff_text: str) -> list[str]:
    paths: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("+++ "):
            continue
        raw_path = line[4:].strip()
        if raw_path == "/dev/null":
            continue
        normalized = raw_path[2:] if raw_path.startswith("b/") else raw_path
        if normalized and normalized not in paths:
            paths.append(normalized)
    return paths


def _path_is_high_risk(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(pattern.search(normalized) for pattern in _HANDOFF_HIGH_RISK_PATTERNS)


def _path_matches_scope(path: str, scopes: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    for scope in scopes:
        normalized_scope = scope.replace("\\", "/").rstrip("/")
        if normalized == normalized_scope or normalized.startswith(f"{normalized_scope}/"):
            return True
    return False


def _artifact_is_raw_diff_fallback(artifact: HandoffArtifact) -> bool:
    return artifact.summary.endswith("returned a raw diff patch.") and artifact.rationale.startswith(
        "The bridge engine returned only a patch"
    )
