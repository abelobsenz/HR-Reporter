from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from app.models import TextChunk


SNAPSHOT_SYSTEM_PROMPT = """
You are an HR diagnostic extraction engine.
Rules:
1) Use only the provided chunk text.
2) Never invent facts or metrics. If a value is not explicit, return null and mark evidence_map status as "not_provided_in_sources".
3) Extract when explicit: if a sentence clearly states a fact, record it even if wording is varied.
   Examples of explicit statements:
   - policy existence: "we have an anti-harassment policy", "our sick time policy states..."
   - system usage: "enter time away in Workday", "we use BambooHR as HRIS"
4) Ignore navigation/TOC/menu labels and index lists as evidence. Short labels alone (e.g., "Time Off", "Job Families") are not sufficient.
5) Return `evidence_map` as an array of items:
   {"field_path": string, "status": "present"|"not_provided_in_sources"|"explicitly_missing"|"not_assessed", "citations": [{"chunk_id": string, "snippet": string}]}
6) For evidence_map citations, always include chunk_id and a short verbatim snippet from the provided chunk.
7) Citations must be sentence-like (not labels): prefer snippets with punctuation and enough context to stand alone.
8) If evidence is missing, use status "not_provided_in_sources" and citations [].
9) Return strict JSON that matches the provided schema.
""".strip()

PLAN_SYSTEM_PROMPT = """
You produce a 30/60/90-day HR action plan.
Rules:
1) Use only snapshot and findings provided.
2) Do not introduce new facts.
3) Every action must include evidence citations from findings. If no evidence exists, use a single citation {"chunk_id":"not_found","snippet":"not found"}.
4) If a step depends on unknown inputs, label it "Conditional: if X is not in place...".
5) Do not assume specific owners unless supported by findings; otherwise use Owner: TBD language.
6) Keep wording client-ready and concise.
7) Return strict JSON matching the schema.
""".strip()


def chunks_to_prompt_text(chunks: List[TextChunk], max_chars_per_chunk: int = 1400) -> str:
    blocks: List[str] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = f"{text[:max_chars_per_chunk]}..."
        blocks.append(
            f"[chunk_id={chunk.chunk_id}]\n"
            f"doc_id={chunk.doc_id}\n"
            f"section={chunk.section}\n"
            f"text:\n{text}"
        )
    return "\n\n".join(blocks)


def build_snapshot_user_prompt(
    *,
    chunks: List[TextChunk],
    tracked_fields: Iterable[str],
    prompt_overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = prompt_overrides or {}
    payload = {
        "task": "Extract CompanyPeopleSnapshot",
        "tracked_fields_for_evidence_map": list(tracked_fields),
        "additional_guidance": overrides.get("snapshot_guidance", ""),
    }
    return (
        f"Instructions:\n{json.dumps(payload, indent=2)}\n\n"
        "Source chunks:\n"
        f"{chunks_to_prompt_text(chunks)}"
    )


def build_plan_user_prompt(
    *,
    stage_label: str,
    snapshot_payload: Dict[str, Any],
    findings_payload: List[Dict[str, Any]],
    prompt_overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = prompt_overrides or {}
    task = {
        "stage_label": stage_label,
        "style": overrides.get("plan_style", "pragmatic, consultant-ready"),
        "snapshot": snapshot_payload,
        "findings": findings_payload,
    }
    return f"Generate plan JSON for this payload:\n{json.dumps(task, indent=2)}"
