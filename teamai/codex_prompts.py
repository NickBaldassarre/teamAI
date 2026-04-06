from __future__ import annotations

from .schemas import CodexHandoffPayload


CODEX_LEAD_ARCHITECT_SYSTEM_PROMPT = (
    "You are the Lead Architect. Do not ask for more file context. "
    "Rely entirely on this distilled context. Output your solution as a standard "
    "unified diff format (.patch) so it can be applied securely to the local repo. "
    "Do not wrap the patch in markdown fences. Do not return prose before or after the patch."
)


def build_codex_handoff_prompt(payload: CodexHandoffPayload) -> str:
    dependencies = "\n".join(f"- {path}" for path in payload.core_dependencies) or "- (none provided)"
    distilled_sections = []
    for path, summary in payload.distilled_context.items():
        distilled_sections.append(f"[{path}]\n{summary}")
    distilled_context = "\n\n".join(distilled_sections) or "(no distilled context provided)"

    return (
        f"{CODEX_LEAD_ARCHITECT_SYSTEM_PROMPT}\n\n"
        f"Original task:\n{payload.original_task}\n\n"
        f"Core dependencies:\n{dependencies}\n\n"
        f"Distilled context:\n{distilled_context}\n\n"
        f"Recommended Codex action:\n{payload.recommended_codex_action}\n\n"
        "Patch output requirements:\n"
        "- Return only a unified diff patch.\n"
        "- Change only the files implied by the distilled context and recommended action.\n"
        "- Preserve existing behavior outside the scoped fix.\n"
        "- Do not ask for additional repository inspection.\n"
        "- If no safe patch is possible, return a minimal patch that adds a TODO comment explaining the blocker."
    )
