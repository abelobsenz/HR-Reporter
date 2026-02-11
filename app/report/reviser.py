from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Tuple

from app.llm.client import OpenAIResponsesClient

logger = logging.getLogger(__name__)

REPORT_REVISER_SYSTEM_PROMPT = """
You revise HR assessment markdown for readability and correctness.
Rules:
1) Keep the exact markdown report structure and section ordering.
2) Do not change factual meaning, numbers, dates, names, risk levels, or owners.
3) Preserve all citations and chunk IDs exactly as written.
4) Improve only wording quality: grammar, clarity, concision, and awkward phrasing.
5) Do not add new claims, recommendations, sources, or legal interpretations.
6) Return markdown only. No code fences, no preface.
""".strip()

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
CHUNK_ID_RE = re.compile(r"\b([A-Za-z0-9_-]+-c\d{3,5})\b")
FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*([\s\S]*?)\s*```\s*$", flags=re.IGNORECASE)


def _heading_signature(markdown: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for raw in (markdown or "").splitlines():
        match = HEADING_RE.match(raw)
        if not match:
            continue
        out.append((len(match.group(1)), match.group(2).strip()))
    return out


def _chunk_ids(markdown: str) -> List[str]:
    out: List[str] = []
    for match in CHUNK_ID_RE.finditer(markdown or ""):
        chunk_id = match.group(1)
        if chunk_id not in out:
            out.append(chunk_id)
    return out


def _unwrap_markdown_fence(text: str) -> str:
    match = FENCE_RE.match(text or "")
    if not match:
        return text
    return match.group(1).strip()


def _build_revision_prompt(*, markdown: str, profile_name: str | None = None) -> str:
    profile_label = (profile_name or "").strip() or "Default Profile"
    return (
        "Task: Revise the final report markdown for clean professional writing while preserving substance.\n"
        f"Profile: {profile_label}\n\n"
        "Return markdown only.\n\n"
        "Input report markdown:\n"
        "```markdown\n"
        f"{markdown}\n"
        "```"
    )


def revise_markdown_report(
    *,
    client: OpenAIResponsesClient,
    markdown: str,
    profile_name: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    source = (markdown or "").strip()
    model = os.getenv("OPENAI_REVISER_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
    if not source:
        return markdown, {
            "enabled": True,
            "attempted": False,
            "applied": False,
            "model": model,
            "reason": "empty_markdown",
        }

    if not client.is_enabled():
        return markdown, {
            "enabled": True,
            "attempted": False,
            "applied": False,
            "model": model,
            "reason": "client_disabled",
        }

    original_headings = _heading_signature(source)
    original_chunk_ids = _chunk_ids(source)

    try:
        revised = client.text_completion(
            system_prompt=REPORT_REVISER_SYSTEM_PROMPT,
            user_prompt=_build_revision_prompt(markdown=source, profile_name=profile_name),
            max_output_tokens=None,
            model_override=model,
            schema_name="report_markdown_reviser",
        )
    except Exception:
        logger.exception("report_reviser_failed")
        return markdown, {
            "enabled": True,
            "attempted": True,
            "applied": False,
            "model": model,
            "reason": "api_error",
        }

    revised_clean = _unwrap_markdown_fence((revised or "").strip())
    if not revised_clean:
        return markdown, {
            "enabled": True,
            "attempted": True,
            "applied": False,
            "model": model,
            "reason": "empty_revision",
        }

    revised_headings = _heading_signature(revised_clean)
    revised_chunk_ids = _chunk_ids(revised_clean)
    heading_changed = revised_headings != original_headings
    chunks_changed = sorted(revised_chunk_ids) != sorted(original_chunk_ids)

    if heading_changed or chunks_changed:
        logger.warning(
            "report_reviser_validation_failed heading_changed=%s chunks_changed=%s",
            heading_changed,
            chunks_changed,
        )
        return markdown, {
            "enabled": True,
            "attempted": True,
            "applied": False,
            "model": model,
            "reason": "validation_failed",
            "validation": {
                "heading_changed": heading_changed,
                "chunks_changed": chunks_changed,
                "original_headings": len(original_headings),
                "revised_headings": len(revised_headings),
                "original_chunk_ids": len(original_chunk_ids),
                "revised_chunk_ids": len(revised_chunk_ids),
            },
        }

    if revised_clean == source:
        return markdown, {
            "enabled": True,
            "attempted": True,
            "applied": False,
            "model": model,
            "reason": "no_changes",
            "validation": {
                "heading_changed": False,
                "chunks_changed": False,
            },
        }

    return revised_clean, {
        "enabled": True,
        "attempted": True,
        "applied": True,
        "model": model,
        "reason": "success",
        "validation": {
            "heading_changed": False,
            "chunks_changed": False,
            "original_headings": len(original_headings),
            "original_chunk_ids": len(original_chunk_ids),
        },
    }
