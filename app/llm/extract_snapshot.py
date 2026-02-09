from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import logging

from app.llm.client import OpenAIResponsesClient
from app.llm.prompts import (
    PLAN_SYSTEM_PROMPT,
    SNAPSHOT_SYSTEM_PROMPT,
    build_plan_user_prompt,
    build_snapshot_user_prompt,
)
from app.models import (
    Citation,
    CompanyPeopleSnapshot,
    EvidenceStatus,
    Finding,
    Plan3090,
    TextChunk,
)
from app.utils import load_json

logger = logging.getLogger(__name__)


BOOLEAN_TRACKED_FIELDS = {
    "policies.employee_handbook",
    "policies.anti_harassment_policy",
    "policies.leave_policy",
    "policies.overtime_policy",
    "policies.accommodation_policy",
    "policies.data_privacy_policy",
    "policies.disciplinary_policy",
    "hiring.structured_interviews",
    "hiring.job_architecture",
    "hiring.background_checks",
    "hiring.offer_approval",
    "onboarding.onboarding_program",
    "onboarding.role_plan_30_60_90",
    "manager_enablement.manager_training",
    "comp_leveling.pay_bands",
    "comp_leveling.leveling_framework",
    "hris_data.mandatory_training_tracking",
    "er_retention.er_case_process",
    "er_retention.engagement_survey",
    "benefits.benefits_broker",
    "benefits.benefits_eligibility_policy",
}

SPECIAL_FIELD_ALIASES = {
    "policies.leave_policy": [
        "leave policy",
        "time off policy",
        "sick time policy",
        "pto policy",
        "paid time off policy",
        "time off and absence policy",
        "absence policy",
    ],
    "hris_data.hris_system": [
        "hris",
        "human resources information system",
        "time away",
        "time off",
        "payroll",
        "onboarding",
    ],
}

HRIS_SYSTEM_NAMES = [
    "Workday",
    "BambooHR",
    "Rippling",
    "ADP",
    "UKG",
    "SuccessFactors",
    "Oracle HCM",
    "Namely",
    "Gusto",
    "Paylocity",
    "Dayforce",
    "SAP SuccessFactors",
]

NEGATIVE_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\bdoes not\b",
    r"\bdon't\b",
    r"\bmissing\b",
    r"\black\b",
]

POSITIVE_PATTERNS = [
    r"\bhas\b",
    r"\bhave\b",
    r"\buses?\b",
    r"\busing\b",
    r"\bin place\b",
    r"\bpolicy\b",
    r"\bdocumented\b",
]


def _schema_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / name


def extract_snapshot(
    *,
    client: OpenAIResponsesClient,
    chunks: List[TextChunk],
    tracked_fields: Iterable[str],
    prompt_overrides: Dict[str, Any] | None = None,
    targeted_retrieval: Dict[str, Any] | None = None,
) -> CompanyPeopleSnapshot:
    tracked_fields_list = list(tracked_fields)
    must_include_chunk_ids = _must_include_chunk_ids_from_retrieval(targeted_retrieval)
    logger.info(
        "snapshot_prepare chunks=%s tracked_fields=%s must_include=%s",
        len(chunks),
        len(tracked_fields_list),
        len(must_include_chunk_ids),
    )
    prepared_chunks = _prepare_snapshot_chunks(
        chunks=chunks,
        tracked_fields=tracked_fields_list,
        max_chunks=max(1, _env_int("HR_REPORT_SNAPSHOT_MAX_CHUNKS", 80)),
        must_include_chunk_ids=must_include_chunk_ids,
    )

    if not chunks:
        snapshot = CompanyPeopleSnapshot()
        for field_path in tracked_fields_list:
            snapshot.evidence_map[field_path] = EvidenceStatus(
                status="not_assessed", citations=[]
            )
        return snapshot

    schema = _snapshot_schema_for_openai()
    max_prompt_chars = max(1, _env_int("HR_REPORT_SNAPSHOT_MAX_PROMPT_CHARS", 55_000))
    estimated_prompt_chars = len(
        build_snapshot_user_prompt(
            chunks=prepared_chunks,
            tracked_fields=tracked_fields_list,
            prompt_overrides=prompt_overrides,
        )
    )
    if estimated_prompt_chars > max_prompt_chars and len(prepared_chunks) > 1:
        batch_size = max(1, _env_int("HR_REPORT_SNAPSHOT_CHUNK_BATCH_SIZE", 24))
        payloads: List[Dict[str, Any]] = []
        chunk_batches = _chunk_batches(prepared_chunks, batch_size)
        logger.info(
            "snapshot_batch_mode enabled=true chunks=%s batches=%s batch_size=%s estimated_prompt_chars=%s",
            len(prepared_chunks),
            len(chunk_batches),
            batch_size,
            estimated_prompt_chars,
        )
        for index, chunk_batch in enumerate(chunk_batches, start=1):
            logger.info(
                "snapshot_batch_start batch=%s/%s chunks=%s",
                index,
                len(chunk_batches),
                len(chunk_batch),
            )
            batch_payload = _extract_snapshot_payload_with_retry(
                client=client,
                chunks=chunk_batch,
                tracked_fields=tracked_fields_list,
                prompt_overrides=prompt_overrides,
                schema=schema,
            )
            batch_payload = _coerce_evidence_map_entries_to_dict(batch_payload)
            batch_payload = _anchor_payload_citations(batch_payload, chunk_batch)
            payloads.append(batch_payload)
            logger.info("snapshot_batch_done batch=%s/%s", index, len(chunk_batches))
        payload = _merge_snapshot_payloads(payloads=payloads, tracked_fields=tracked_fields_list)
    else:
        logger.info(
            "snapshot_batch_mode enabled=false chunks=%s estimated_prompt_chars=%s",
            len(prepared_chunks),
            estimated_prompt_chars,
        )
        payload = _extract_snapshot_payload_with_retry(
            client=client,
            chunks=prepared_chunks,
            tracked_fields=tracked_fields_list,
            prompt_overrides=prompt_overrides,
            schema=schema,
        )
        payload = _coerce_evidence_map_entries_to_dict(payload)
        payload = _anchor_payload_citations(payload, prepared_chunks)
    snapshot = CompanyPeopleSnapshot.model_validate(payload)
    snapshot = _apply_explicit_chunk_supplements(
        snapshot=snapshot,
        chunks=prepared_chunks,
        tracked_fields=tracked_fields_list,
    )
    logger.info(
        "snapshot_supplements_applied tracked_fields=%s",
        len(tracked_fields_list),
    )
    snapshot = _apply_targeted_retrieval_overlay(
        snapshot=snapshot,
        targeted_retrieval=targeted_retrieval,
        chunks=prepared_chunks,
        tracked_fields=tracked_fields_list,
    )
    logger.info("snapshot_overlay_applied")

    # Ensure every tracked field exists in evidence_map for deterministic downstream checks.
    for field_path in tracked_fields_list:
        if field_path not in snapshot.evidence_map:
            snapshot.evidence_map[field_path] = EvidenceStatus(
                status="not_provided_in_sources", citations=[]
            )

    return snapshot


def _snapshot_schema_for_openai() -> Dict[str, Any]:
    """Build OpenAI-compatible snapshot schema.

    The persisted app schema keeps `evidence_map` as an object map. OpenAI strict
    schemas are more reliable with arrays than map-like `additionalProperties`.
    """
    schema = load_json(_schema_path("snapshot_schema.json"))
    properties = schema.get("properties", {})
    properties["evidence_map"] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "field_path": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": [
                        "present",
                        "not_provided_in_sources",
                        "explicitly_missing",
                        "not_assessed",
                    ],
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                        "required": ["chunk_id", "snippet"],
                    },
                },
            },
            "required": ["field_path", "status", "citations"],
        },
    }
    schema["required"] = list(properties.keys())
    return schema


def _coerce_evidence_map_entries_to_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_evidence_map = payload.get("evidence_map", [])
    if isinstance(raw_evidence_map, list):
        evidence_map: Dict[str, Dict[str, Any]] = {}
        for entry in raw_evidence_map:
            if not isinstance(entry, dict):
                continue
            field_path = entry.get("field_path")
            if not isinstance(field_path, str) or not field_path:
                continue
            status = _normalize_evidence_status(entry.get("status"))
            citations = _filter_sentence_like_citations(entry.get("citations", []))
            if status in {"present", "explicitly_missing"} and len(citations) == 0:
                status = "not_provided_in_sources"
            if status == "not_provided_in_sources":
                citations = []
            evidence_map[field_path] = {
                "status": status,
                "retrieval_status": None,
                "coverage_note": None,
                "retrieval_queries": [],
                "match_count": len(citations),
                "citations": citations,
            }
        payload["evidence_map"] = evidence_map
    return payload


def _normalize_evidence_status(raw: Any) -> str:
    if raw in {
        "present",
        "not_provided_in_sources",
        "explicitly_missing",
        "not_assessed",
    }:
        return str(raw)
    return "not_provided_in_sources"


def _field_aliases(field_path: str) -> List[str]:
    aliases = list(SPECIAL_FIELD_ALIASES.get(field_path, []))
    leaf = field_path.split(".")[-1].replace("_", " ").strip().lower()
    if leaf and leaf not in aliases:
        aliases.append(leaf)
    return aliases


def _split_sentences(text: str) -> List[str]:
    raw = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    out: List[str] = []
    for sentence in raw:
        cleaned = " ".join(sentence.split()).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _is_nav_like_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    short_line_count = sum(1 for line in lines if len(line.split()) <= 4)
    punct_line_count = sum(1 for line in lines if any(token in line for token in ".:;!?"))
    return (short_line_count / len(lines) >= 0.6) and (punct_line_count / len(lines) <= 0.25)


def _chunk_signal_score(chunk: TextChunk, tracked_fields: List[str]) -> int:
    text = chunk.text
    lower = text.lower()
    score = 0

    if len(text) >= 320:
        score += 1
    if len(_split_sentences(text)) >= 2:
        score += 2
    if _is_nav_like_text(text):
        score -= 4

    keyword_hits = 0
    for field in tracked_fields:
        for alias in _field_aliases(field):
            if alias and alias in lower:
                keyword_hits += 1
                break
    if keyword_hits > 0:
        score += min(3, keyword_hits)

    if any(re.search(pattern, lower) for pattern in POSITIVE_PATTERNS + NEGATIVE_PATTERNS):
        score += 1

    return score


def _compress_chunk_text(chunk: TextChunk, tracked_fields: List[str]) -> str:
    sentences = _split_sentences(chunk.text)
    if not sentences:
        return chunk.text

    keywords: List[str] = []
    for field in tracked_fields:
        keywords.extend(_field_aliases(field))
    keyword_set = {k for k in keywords if k}

    selected: List[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        has_keyword = any(keyword in lower for keyword in keyword_set)
        has_signal = any(re.search(pattern, lower) for pattern in POSITIVE_PATTERNS + NEGATIVE_PATTERNS)
        if has_keyword or has_signal:
            selected.append(sentence)
        if len(selected) >= 6:
            break

    if len(selected) < 2:
        selected = sentences[: min(4, len(sentences))]

    compressed = " ".join(selected).strip()
    return compressed or chunk.text


def _prepare_snapshot_chunks(
    *,
    chunks: List[TextChunk],
    tracked_fields: List[str],
    max_chunks: int,
    must_include_chunk_ids: set[str] | None = None,
) -> List[TextChunk]:
    if not chunks:
        return chunks

    deduped: List[TextChunk] = []
    seen = set()
    for chunk in chunks:
        key = " ".join(chunk.text.split()).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    scored: List[Tuple[int, TextChunk]] = []
    for chunk in deduped:
        score = _chunk_signal_score(chunk, tracked_fields)
        scored.append((score, chunk))

    scored.sort(key=lambda item: (item[0], len(item[1].text)), reverse=True)
    must_include = must_include_chunk_ids or set()
    must_chunks = [chunk for _, chunk in scored if chunk.chunk_id in must_include]
    kept = [chunk for score, chunk in scored if score >= 0 and chunk.chunk_id not in must_include]
    kept = must_chunks + kept
    kept = kept[:max_chunks]
    if not kept:
        kept = deduped[:max_chunks]

    prepared: List[TextChunk] = []
    for chunk in kept:
        prepared.append(
            TextChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                section=chunk.section,
                heading_path=list(chunk.heading_path),
                is_list=bool(chunk.is_list),
                is_table=bool(chunk.is_table),
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                nav_score=chunk.nav_score,
                text=_compress_chunk_text(chunk, tracked_fields),
            )
        )
    return prepared


def _is_sentence_like_citation(snippet: str) -> bool:
    text = " ".join(str(snippet).split()).strip()
    if len(text) < 60:
        return False
    if not any(token in text for token in [".", ":", ";"]):
        return False
    return True


def _filter_sentence_like_citations(raw_citations: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_citations, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        if not chunk_id or not snippet:
            continue
        if not _is_sentence_like_citation(snippet):
            continue
        out.append({"chunk_id": chunk_id, "snippet": snippet})
    return out


def _find_best_boolean_evidence(chunk: TextChunk, aliases: List[str]) -> Tuple[bool | None, Dict[str, str] | None]:
    for sentence in _split_sentences(chunk.text):
        lower = sentence.lower()
        if not any(alias in lower for alias in aliases if alias):
            continue
        if not _is_sentence_like_citation(sentence):
            continue
        if any(re.search(pattern, lower) for pattern in NEGATIVE_PATTERNS):
            return False, {"chunk_id": chunk.chunk_id, "snippet": sentence}
        if any(re.search(pattern, lower) for pattern in POSITIVE_PATTERNS):
            return True, {"chunk_id": chunk.chunk_id, "snippet": sentence}
    return None, None


def _find_hris_system_evidence(chunk: TextChunk) -> Tuple[str | None, Dict[str, str] | None]:
    for sentence in _split_sentences(chunk.text):
        lower = sentence.lower()
        if not _is_sentence_like_citation(sentence):
            continue
        has_context = any(term in lower for term in SPECIAL_FIELD_ALIASES["hris_data.hris_system"])
        if not has_context:
            continue
        for system in HRIS_SYSTEM_NAMES:
            if re.search(rf"\b{re.escape(system.lower())}\b", lower):
                return system, {"chunk_id": chunk.chunk_id, "snippet": sentence}
    return None, None


def _apply_explicit_chunk_supplements(
    *,
    snapshot: CompanyPeopleSnapshot,
    chunks: List[TextChunk],
    tracked_fields: List[str],
) -> CompanyPeopleSnapshot:
    payload = snapshot.model_dump(mode="python")
    evidence_map = dict(payload.get("evidence_map", {}))

    for field_path in tracked_fields:
        current_value = _get_path(payload, field_path)
        current_evidence = evidence_map.get(field_path, {})
        current_status = _normalize_evidence_status(current_evidence.get("status"))

        if current_value is not None:
            continue
        if current_status in {"present", "explicitly_missing"}:
            continue

        # Explicit HRIS name extraction from sentence-level context.
        if field_path == "hris_data.hris_system":
            for chunk in chunks:
                system, citation = _find_hris_system_evidence(chunk)
                if not system or not citation:
                    continue
                _set_path(payload, field_path, system)
                evidence_map[field_path] = {"status": "present", "citations": [citation]}
                break
            continue

        if field_path not in BOOLEAN_TRACKED_FIELDS:
            continue

        aliases = _field_aliases(field_path)
        for chunk in chunks:
            value, citation = _find_best_boolean_evidence(chunk, aliases)
            if value is None or citation is None:
                continue
            _set_path(payload, field_path, value)
            evidence_map[field_path] = {"status": "present", "citations": [citation]}
            break

    payload["evidence_map"] = evidence_map
    return CompanyPeopleSnapshot.model_validate(payload)


def _must_include_chunk_ids_from_retrieval(targeted_retrieval: Dict[str, Any] | None) -> set[str]:
    if not isinstance(targeted_retrieval, dict):
        return set()
    out = set()
    for candidates in targeted_retrieval.get("field_candidates", {}).values():
        if not isinstance(candidates, list):
            continue
        for candidate in candidates[:2]:
            if not isinstance(candidate, dict):
                continue
            chunk_id = candidate.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                out.add(chunk_id)
    return out


def _anchor_payload_citations(payload: Dict[str, Any], chunks: List[TextChunk]) -> Dict[str, Any]:
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    evidence_map = payload.get("evidence_map")
    if not isinstance(evidence_map, dict):
        return payload
    for field_path, evidence in evidence_map.items():
        if not isinstance(evidence, dict):
            continue
        anchored = []
        for citation in evidence.get("citations", []):
            if not isinstance(citation, dict):
                continue
            chunk_id = str(citation.get("chunk_id", "")).strip()
            snippet = str(citation.get("snippet", "")).strip()
            if not chunk_id or not snippet:
                continue
            chunk = chunk_map.get(chunk_id)
            citation_payload = {
                "chunk_id": chunk_id,
                "snippet": snippet,
                "evidence_recovered_by": "llm",
            }
            if chunk is not None:
                citation_payload["doc_id"] = chunk.doc_id
                citation_payload["start_char"] = chunk.start_char
                citation_payload["end_char"] = chunk.end_char
            anchored.append(citation_payload)
        evidence["citations"] = anchored
        evidence["match_count"] = len(anchored)
        evidence_map[field_path] = evidence
    payload["evidence_map"] = evidence_map
    return payload


def _apply_targeted_retrieval_overlay(
    *,
    snapshot: CompanyPeopleSnapshot,
    targeted_retrieval: Dict[str, Any] | None,
    chunks: List[TextChunk],
    tracked_fields: List[str],
) -> CompanyPeopleSnapshot:
    if not isinstance(targeted_retrieval, dict):
        return snapshot

    payload = snapshot.model_dump(mode="python")
    evidence_map = dict(payload.get("evidence_map", {}))
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

    field_statuses = targeted_retrieval.get("field_statuses", {})
    field_candidates = targeted_retrieval.get("field_candidates", {})
    coverage_notes = targeted_retrieval.get("coverage_notes", {})
    queries_by_field = targeted_retrieval.get("queries_by_field", {})

    for field_path in tracked_fields:
        status_guess = field_statuses.get(field_path)
        candidates = field_candidates.get(field_path, []) or []
        evidence = evidence_map.get(field_path, {})
        if not isinstance(evidence, dict):
            evidence = {"status": "not_provided_in_sources", "citations": []}

        evidence["retrieval_status"] = status_guess
        evidence["coverage_note"] = coverage_notes.get(field_path)
        evidence["retrieval_queries"] = list(queries_by_field.get(field_path, []))[:12]
        evidence["match_count"] = len(candidates)

        # Recovery rule: use top explicit candidate when LLM missed field.
        def _has_anchor(cits: Any) -> bool:
            if not isinstance(cits, list) or not cits:
                return False
            first = cits[0] if isinstance(cits[0], dict) else {}
            if not isinstance(first, dict):
                return False
            return bool(first.get("doc_id")) and (first.get("start_char") is not None)

        def _top_candidate_citation() -> Dict[str, Any] | None:
            if not candidates:
                return None
            top = candidates[0]
            chunk = chunk_map.get(str(top.get("chunk_id", "")))
            citation = {
                "chunk_id": str(top.get("chunk_id", "")),
                "snippet": str(top.get("snippet", "")),
                "evidence_recovered_by": "targeted_retrieval",
                "retrieval_score": float(top.get("score", 0.0)),
            }
            if chunk is not None:
                citation["doc_id"] = chunk.doc_id
                citation["start_char"] = chunk.start_char
                citation["end_char"] = chunk.end_char
            if not citation["chunk_id"] or not citation["snippet"]:
                return None
            return citation

        if (
            status_guess == "MENTIONED_EXPLICIT"
            and (evidence.get("status") != "present" or not _has_anchor(evidence.get("citations", [])))
            and candidates
        ):
            citation = _top_candidate_citation()
            if citation is not None:
                evidence["status"] = "present"
                evidence["citations"] = [citation]
                evidence["match_count"] = max(1, len(candidates))

        if (
            status_guess in {"MENTIONED_IMPLICIT", "MENTIONED_AMBIGUOUS"}
            and (evidence.get("status") != "present" or not _has_anchor(evidence.get("citations", [])))
            and evidence.get("status") != "explicitly_missing"
            and candidates
        ):
            citation = _top_candidate_citation()
            if citation is not None:
                # Preserve context for ambiguous/implicit retrieval so downstream findings are auditable.
                if evidence.get("status") not in {"present", "explicitly_missing"}:
                    evidence["status"] = "not_provided_in_sources"
                evidence["citations"] = [citation]
                evidence["match_count"] = max(1, len(candidates))

        if status_guess in {"NOT_RETRIEVED", "NOT_FOUND_IN_RETRIEVED"}:
            if evidence.get("status") not in {"present", "explicitly_missing"}:
                evidence["status"] = "not_provided_in_sources"
                if status_guess == "NOT_RETRIEVED":
                    evidence["citations"] = []

        evidence_map[field_path] = evidence

    payload["evidence_map"] = evidence_map
    return CompanyPeopleSnapshot.model_validate(payload)


def generate_plan(
    *,
    client: OpenAIResponsesClient,
    stage_label: str,
    snapshot: CompanyPeopleSnapshot,
    findings: List[Finding],
    prompt_overrides: Dict[str, Any] | None = None,
) -> Plan3090:
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "why_now": {"type": "string"},
            "days_30": {"type": "array", "items": _plan_action_schema()},
            "days_60": {"type": "array", "items": _plan_action_schema()},
            "days_90": {"type": "array", "items": _plan_action_schema()},
        },
        "required": ["why_now", "days_30", "days_60", "days_90"],
    }
    payload = client.structured_json(
        system_prompt=PLAN_SYSTEM_PROMPT,
        user_prompt=build_plan_user_prompt(
            stage_label=stage_label,
            snapshot_payload=snapshot.model_dump(mode="json"),
            findings_payload=[finding.model_dump(mode="json") for finding in findings],
            prompt_overrides=prompt_overrides,
        ),
        json_schema=schema,
        schema_name="plan_30_60_90",
    )
    return Plan3090.model_validate(payload)


def deterministic_plan_from_findings(findings: List[Finding]) -> Plan3090:
    default_citation = [Citation(chunk_id="not_found", snippet="not found")]

    def _action_from_finding(finding: Finding, suffix: str) -> Dict[str, Any]:
        evidence = finding.evidence or default_citation
        prefix = "Conditional: " if finding.needs_confirmation else ""
        return {
            "action": f"{prefix}{finding.title}: {suffix}",
            "rationale": f"{finding.stage_reason} {finding.owner}".strip(),
            "evidence": [citation.model_dump() for citation in evidence],
        }

    top = findings[:6]
    actions_30 = [_action_from_finding(f, "define owner and scope") for f in top[:2]]
    actions_60 = [_action_from_finding(f, "pilot and iterate") for f in top[2:4]]
    actions_90 = [_action_from_finding(f, "standardize and track") for f in top[4:6]]

    if not actions_30:
        actions_30 = [
            {
                "action": "Validate baseline HR controls",
                "rationale": "No high-priority gaps were detected; confirm controls remain effective.",
                "evidence": [citation.model_dump() for citation in default_citation],
            }
        ]

    return Plan3090.model_validate(
        {
            "why_now": "Sequencing core HR controls now reduces scale-related execution and compliance risk.",
            "days_30": actions_30,
            "days_60": actions_60,
            "days_90": actions_90,
        }
    )


def _plan_action_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string"},
            "rationale": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["chunk_id", "snippet"],
                },
            },
        },
        "required": ["action", "rationale", "evidence"],
    }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _chunk_batches(items: List[TextChunk], size: int) -> List[List[TextChunk]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_snapshot_payload_with_retry(
    *,
    client: OpenAIResponsesClient,
    chunks: List[TextChunk],
    tracked_fields: List[str],
    prompt_overrides: Dict[str, Any] | None,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return _extract_snapshot_payload_once(
            client=client,
            chunks=chunks,
            tracked_fields=tracked_fields,
            prompt_overrides=prompt_overrides,
            schema=schema,
        )
    except Exception as exc:
        if not _is_context_length_error(exc) or len(chunks) <= 1:
            raise
        midpoint = len(chunks) // 2
        logger.warning(
            "snapshot_context_split triggered=true chunks=%s left=%s right=%s",
            len(chunks),
            midpoint,
            len(chunks) - midpoint,
        )
        left = _extract_snapshot_payload_with_retry(
            client=client,
            chunks=chunks[:midpoint],
            tracked_fields=tracked_fields,
            prompt_overrides=prompt_overrides,
            schema=schema,
        )
        right = _extract_snapshot_payload_with_retry(
            client=client,
            chunks=chunks[midpoint:],
            tracked_fields=tracked_fields,
            prompt_overrides=prompt_overrides,
            schema=schema,
        )
        return _merge_snapshot_payloads(
            payloads=[left, right],
            tracked_fields=tracked_fields,
        )


def _extract_snapshot_payload_once(
    *,
    client: OpenAIResponsesClient,
    chunks: List[TextChunk],
    tracked_fields: List[str],
    prompt_overrides: Dict[str, Any] | None,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    return client.structured_json(
        system_prompt=SNAPSHOT_SYSTEM_PROMPT,
        user_prompt=build_snapshot_user_prompt(
            chunks=chunks,
            tracked_fields=tracked_fields,
            prompt_overrides=prompt_overrides,
        ),
        json_schema=schema,
        schema_name="company_people_snapshot",
    )


def _is_context_length_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "context window" in message
        or "context_length_exceeded" in message
        or "input exceeds" in message
    )


def _merge_snapshot_payloads(
    *,
    payloads: List[Dict[str, Any]],
    tracked_fields: List[str],
) -> Dict[str, Any]:
    merged = CompanyPeopleSnapshot().model_dump(mode="python")
    sources_by_field: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}

    for idx, payload in enumerate(payloads):
        payload = _coerce_evidence_map_entries_to_dict(dict(payload))
        evidence_map = payload.get("evidence_map", {}) or {}
        for field_path, raw_status in evidence_map.items():
            if not isinstance(raw_status, dict):
                continue
            status = _normalize_evidence_status(raw_status.get("status"))
            citations = raw_status.get("citations", [])
            rank = _evidence_rank(status)
            current = sources_by_field.get(field_path)
            candidate = (rank, len(citations), {"status": status, "citations": citations, "idx": idx})
            if current is None or (candidate[0], candidate[1]) > (current[0], current[1]):
                sources_by_field[field_path] = candidate

    # Merge top-level list fields across batches.
    for list_field in ["primary_locations", "current_priorities", "key_risks"]:
        values: List[Any] = []
        for payload in payloads:
            for item in payload.get(list_field, []) or []:
                if item not in values:
                    values.append(item)
        merged[list_field] = values

    # Apply best evidence/value per tracked field.
    merged_evidence_map: Dict[str, Dict[str, Any]] = {}
    for field_path in tracked_fields:
        source = sources_by_field.get(field_path)
        if source is None:
            merged_evidence_map[field_path] = {
                "status": "not_provided_in_sources",
                "citations": [],
            }
            continue

        _, _, info = source
        status = str(info["status"])
        citations = _dedupe_citations(info.get("citations", []))
        merged_evidence_map[field_path] = {"status": status, "citations": citations}

        value = _get_path(payloads[int(info["idx"])], field_path)
        if value is not None and value != []:
            _set_path(merged, field_path, value)

    # Backfill a few useful scalar fields if they were not included in tracked_fields.
    for scalar in ["company_name", "headcount", "headcount_range"]:
        if merged.get(scalar) is not None:
            continue
        for payload in payloads:
            value = payload.get(scalar)
            if value is not None:
                merged[scalar] = value
                break

    merged["evidence_map"] = merged_evidence_map
    return merged


def _evidence_rank(status: str) -> int:
    order = {
        "not_provided_in_sources": 1,
        "not_assessed": 2,
        "present": 3,
        "explicitly_missing": 4,
    }
    return order.get(status, 0)


def _dedupe_citations(citations: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(citations, list):
        return out
    for item in citations:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        if not chunk_id or not snippet:
            continue
        if not _is_sentence_like_citation(snippet):
            continue
        key = (chunk_id, snippet)
        if key in seen:
            continue
        seen.add(key)
        payload: Dict[str, Any] = {"chunk_id": chunk_id, "snippet": snippet}
        for optional_key in [
            "source_id",
            "doc_id",
            "start_char",
            "end_char",
            "evidence_recovered_by",
            "retrieval_score",
        ]:
            if optional_key in item and item.get(optional_key) is not None:
                payload[optional_key] = item.get(optional_key)
        out.append(payload)
    return out


def _get_path(payload: Dict[str, Any], field_path: str) -> Any:
    cursor: Any = payload
    for part in field_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _set_path(payload: Dict[str, Any], field_path: str, value: Any) -> None:
    parts = field_path.split(".")
    cursor: Any = payload
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            return
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    if isinstance(cursor, dict):
        cursor[parts[-1]] = value
