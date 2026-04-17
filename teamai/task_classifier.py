"""
Task classification heuristics for teamAI routing.

Extracted from supervisor.py to keep the routing signals centralized
and independently testable.  Each classifier is a pure function that
examines the task string and returns a boolean flag used to populate
``TaskSignature`` fields in the agent registry's scoring pipeline.

Usage::

    from teamai.task_classifier import (
        is_repository_inspection_task,
        is_broad_coding_task,
        is_desktop_task,
        is_explicit_write_task,
    )
"""
from __future__ import annotations

import re


def is_explicit_write_task(task: str) -> bool:
    """True when the task is a focused, file-targeted write request.

    Looks for a write verb (update/edit/modify/...) paired with a
    file-path-like token, or explicit workspace-write API markers.
    """
    text = task.lower()
    has_write_verb = bool(
        re.search(r"\b(update|edit|modify|replace|rewrite|append|insert|write|add)\b", text)
    )
    has_file_target = bool(re.search(r"\b[\w./-]+\.\w+\b", text))
    write_markers = [
        "workspace_write",
        "patch approval",
        "replace_in_file",
        "write_file",
    ]
    return (has_write_verb and has_file_target) or any(marker in text for marker in write_markers)


def is_repository_inspection_task(task: str) -> bool:
    """True when the task is a read-only repo analysis or reconnaissance request."""
    text = task.lower()
    if is_explicit_write_task(text):
        return False
    inspection_markers = ["inspect", "review", "analyze", "understand", "summarize"]
    repo_markers = ["repository", "repo", "codebase", "project", "workspace"]
    task_markers = ["engineering task", "implementation step", "next step", "project state"]
    return (
        any(marker in text for marker in inspection_markers)
        and (any(marker in text for marker in repo_markers) or any(marker in text for marker in task_markers))
    )


def is_broad_coding_task(task: str) -> bool:
    """True when the task describes a broad implementation that exceeds
    what a small local model can do reliably — i.e., it should be routed
    to a verified handoff rather than attempted locally.
    """
    text = task.lower()
    if is_explicit_write_task(text):
        return False
    if is_repository_inspection_task(text):
        return False
    return bool(
        re.search(
            (
                r"\b("
                r"implement|fix|debug|refactor|build|create|add support|wire up|make .* work|ship|complete|"
                r"improve|harden|optimize|extend|upgrade|stabilize|tighten"
                r")\b"
            ),
            text,
        )
    )


def is_desktop_task(task: str) -> bool:
    """True when the task requires OS-level desktop interaction
    (screenshots, clicks, browser automation, etc.).
    """
    text = task.lower()
    return any(
        marker in text
        for marker in (
            "desktop",
            "browser",
            "screenshot",
            "click",
            "open safari",
            "open chrome",
            "computer use",
            "ui automation",
        )
    )
