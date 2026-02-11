from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from app.models import TextChunk


SNAPSHOT_SYSTEM_PROMPT = """
You extract HR facts into the provided JSON schema.
Rules:
1) Use only the provided source chunks or curated evidence payload. Never use outside knowledge.
2) Never invent facts. If a value is not explicitly stated, set it to null.
3) Citations must be copied verbatim from source text and be either:
   - 1-2 complete, standalone sentences, OR
   - a single bullet/list item or a single table row when it is a self-contained policy statement
     (for example: quantity/money/duration/eligibility rules, or clear directives like "Employees may...").
   Do not cite menus/navigation/TOC fragments, and do not cite headings alone.
   If a heading provides context (for example "Bereavement Leave"), cite a bullet/table row under it instead.
4) Evidence map:
   - If evidence supports the field: status "present" and include citations.
   - If evidence explicitly says it is missing: status "explicitly_missing" and include citations.
   - Otherwise: status "not_provided_in_sources" with citations [].
5) Return strict JSON matching the schema.
6) Assessment context is not evidence; use it only to interpret fields.
""".strip()

SNAPSHOT_ANALYST_SYSTEM_PROMPT = """
You are an HR evidence analyst.
Rules:
1) Use only the provided evidence snippets; never use outside knowledge.
2) Evaluate each field independently in HR-native language.
3) Prefer explicit policy/process statements over cultural or role-level statements.
4) If evidence conflicts, mark conflict; do not force a winner.
5) If evidence is weak/indirect, mark ambiguous and set needs_confirmation=true.
6) Keep reasoning concise and practical.
7) Return strict JSON matching the schema.
""".strip()

SNAPSHOT_RESOLVER_SYSTEM_PROMPT = """
You convert analyst judgments into final CompanyPeopleSnapshot JSON.
Rules:
1) Use only analyst judgments and provided evidence.
2) Apply conservative resolution:
   - Set scalar values only for explicit, high-confidence support.
   - Keep null when ambiguous/conflicting/weak.
3) Preserve evidence_map citations and statuses.
4) Do not invent values.
5) Return strict JSON matching the schema.
""".strip()

DISCOVERY_SYSTEM_PROMPT = """
You synthesize additional HR observations not fully covered by the rubric.
Rules:
1) Use only provided evidence snippets.
2) Every observation must cite at least one snippet id, using complete standalone sentence evidence.
3) Do not infer missing controls from generic values/culture statements.
4) If evidence is weak or indirect, label it as a hypothesis and include a concrete follow-up question.
5) Keep findings concise, practical, and stage-aware.
6) Prefer observations that add signal beyond rubric fields already extracted.
7) Return strict JSON matching schema.
8) Assessment context is not evidence.
""".strip()


def _compact_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _fallback_doc_title(chunk: TextChunk) -> str:
    if chunk.doc_title and chunk.doc_title.strip():
        return chunk.doc_title.strip()
    stem = chunk.doc_id.rsplit("-", 1)[0] if "-" in chunk.doc_id else chunk.doc_id
    return stem.replace("_", " ").replace("-", " ").strip() or chunk.doc_id


def _short_source(value: str | None, max_len: int = 96) -> str:
    raw = (value or "").strip()
    if not raw:
        return "unknown"
    if len(raw) <= max_len:
        return raw
    return f"...{raw[-(max_len - 3):]}"


def chunks_to_prompt_text(chunks: List[TextChunk], max_chars_per_chunk: int = 1400) -> str:
    blocks: List[str] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = f"{text[:max_chars_per_chunk]}..."
        heading_path = " > ".join([part.strip() for part in chunk.heading_path if part and part.strip()]) or chunk.section
        blocks.append(
            f"[chunk_id={chunk.chunk_id}]\n"
            f"doc_id={chunk.doc_id}\n"
            f"doc_title={_fallback_doc_title(chunk)}\n"
            f"source={_short_source(chunk.source)}\n"
            f"section={chunk.section}\n"
            f"heading_path={heading_path}\n"
            f"text:\n{text}"
        )
    return "\n\n".join(blocks)


def build_snapshot_user_prompt(
    *,
    chunks: List[TextChunk],
    tracked_fields: Iterable[str],
    assessment_context: str | None = None,
    prompt_overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = prompt_overrides or {}
    payload = {
        "task": "Extract CompanyPeopleSnapshot",
        "tracked_fields_for_evidence_map": list(tracked_fields),
        "assessment_context": assessment_context or "",
        "citation_rule": (
            "Citations must be copied verbatim and can be complete sentences OR one self-contained bullet/table row "
            "that states policy/eligibility/duration/quantity rules."
        ),
        "additional_guidance": overrides.get("snapshot_guidance", ""),
    }
    return (
        f"Instructions:\n{json.dumps(payload, indent=2)}\n\n"
        "Source chunks:\n"
        f"{chunks_to_prompt_text(chunks)}"
    )


def build_snapshot_evidence_prompt(
    *,
    tracked_fields: Iterable[str],
    evidence_by_field: Dict[str, List[Dict[str, Any]]],
    company_context: str | None = None,
    assessment_context: str | None = None,
    prompt_overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = prompt_overrides or {}
    payload = {
        "task": "Extract CompanyPeopleSnapshot from curated evidence",
        "tracked_fields_for_evidence_map": list(tracked_fields),
        "company_context": company_context or "",
        "assessment_context": assessment_context or "",
        "additional_guidance": overrides.get("snapshot_guidance", ""),
        "evidence_by_field": evidence_by_field,
    }
    return f"Evidence payload:\n{_compact_json(payload)}"


def build_snapshot_analysis_prompt(
    *,
    tracked_fields: Iterable[str],
    evidence_by_field: Dict[str, List[Dict[str, Any]]],
    retrieval_statuses: Dict[str, str] | None = None,
    retrieval_queries: Dict[str, List[str]] | None = None,
    company_context: str | None = None,
    assessment_context: str | None = None,
    prompt_overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = prompt_overrides or {}
    payload = {
        "task": "Analyze HR evidence quality and field-level support",
        "tracked_fields": list(tracked_fields),
        "company_context": company_context or "",
        "assessment_context": assessment_context or "",
        "retrieval_statuses": retrieval_statuses or {},
        "retrieval_queries": retrieval_queries or {},
        "additional_guidance": overrides.get("snapshot_guidance", ""),
        "evidence_by_field": evidence_by_field,
    }
    return f"Evidence analysis payload:\n{_compact_json(payload)}"


def build_snapshot_resolver_prompt(
    *,
    tracked_fields: Iterable[str],
    field_assessments: List[Dict[str, Any]],
    company_context: str | None = None,
    assessment_context: str | None = None,
    prompt_overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = prompt_overrides or {}
    payload = {
        "task": "Resolve analyst judgments into CompanyPeopleSnapshot",
        "tracked_fields": list(tracked_fields),
        "company_context": company_context or "",
        "assessment_context": assessment_context or "",
        "additional_guidance": overrides.get("snapshot_guidance", ""),
        "field_assessments": field_assessments,
        "note": "Field assessments already include citations; do not request additional evidence context.",
    }
    return f"Snapshot resolver payload:\n{_compact_json(payload)}"


def build_discovery_user_prompt(
    *,
    stage_label: str,
    evidence_catalog: List[Dict[str, Any]],
    max_findings: int,
    assessment_context: str | None = None,
    prompt_overrides: Dict[str, Any] | None = None,
) -> str:
    overrides = prompt_overrides or {}
    payload = {
        "task": "Identify additional HR observations not fully covered by rubric expectations.",
        "stage_label": stage_label,
        "max_findings": max_findings,
        "assessment_context": assessment_context or "",
        "novelty_rule": "Prefer observations that add signal beyond fields already covered by rubric extraction.",
        "style": overrides.get("synthesis_style", "direct, evidence-grounded"),
        "evidence_catalog": evidence_catalog,
    }
    return f"Generate observation JSON:\n{_compact_json(payload)}"
