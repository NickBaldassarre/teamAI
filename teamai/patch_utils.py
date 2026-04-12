from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchTarget:
    before_path: str | None
    after_path: str | None

    @property
    def primary_path(self) -> str:
        return self.after_path or self.before_path or ""


def extract_patch_targets(patch_text: str) -> list[PatchTarget]:
    targets: list[PatchTarget] = []
    seen: set[tuple[str | None, str | None]] = set()
    pending_before: str | None = None

    for raw_line in patch_text.splitlines():
        if raw_line.startswith("diff --git "):
            pending_before = None
            continue

        if raw_line.startswith("--- "):
            pending_before = _normalize_patch_header_path(raw_line[4:])
            continue

        if raw_line.startswith("+++ "):
            pending_after = _normalize_patch_header_path(raw_line[4:])
            key = (pending_before, pending_after)
            if key not in seen and (pending_before is not None or pending_after is not None):
                seen.add(key)
                targets.append(
                    PatchTarget(
                        before_path=pending_before,
                        after_path=pending_after,
                    )
                )
            pending_before = None

    return targets


def _normalize_patch_header_path(raw_path: str) -> str | None:
    token = raw_path.strip().split("\t", maxsplit=1)[0].strip()
    if not token:
        return None
    if token == "/dev/null":
        return None
    if token.startswith(("a/", "b/")):
        return token[2:]
    return token
