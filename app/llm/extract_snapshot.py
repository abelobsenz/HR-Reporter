from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import logging

from app.llm.client import OpenAIResponsesClient
from app.llm.prompts import (
    SNAPSHOT_ANALYST_SYSTEM_PROMPT,
    SNAPSHOT_RESOLVER_SYSTEM_PROMPT,
    SNAPSHOT_SYSTEM_PROMPT,
    build_snapshot_analysis_prompt,
    build_snapshot_evidence_prompt,
    build_snapshot_resolver_prompt,
    build_snapshot_user_prompt,
)
from app.models import (
    Citation,
    CompanyPeopleSnapshot,
    EvidenceStatus,
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

HEADCOUNT_RANGE_OR_PLUS_RE = re.compile(r"\b\d{1,5}\s*(?:-|to)\s*\d{1,5}\b|\b\d{1,5}\s*\+")
HEADCOUNT_INEXACT_RE = re.compile(
    r"\b(about|around|approximately|approx\.?|~|over|under|nearly|almost|more than|less than|at least|at most)\b"
)
HEADCOUNT_EXACT_PATTERNS: List[tuple[re.Pattern[str], int]] = [
    (
        re.compile(r"\bheadcount\s*(?:is|=|:|at|of)?\s*(\d{1,5})\b", flags=re.IGNORECASE),
        6,
    ),
    (
        re.compile(
            r"\b(?:we|company)\s+(?:have|has|are)\s+(\d{1,5})\s*(employees|employee|fte|team members|people)\b",
            flags=re.IGNORECASE,
        ),
        6,
    ),
    (
        re.compile(r"\b(\d{1,5})\s*(employees|employee|fte|team members)\b", flags=re.IGNORECASE),
        5,
    ),
    (
        re.compile(r"\b(?:team|workforce)\s+of\s+(\d{1,5})\b", flags=re.IGNORECASE),
        4,
    ),
]


def _schema_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / name


def _call_structured_json(
    client: OpenAIResponsesClient,
    *,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    schema_name: str,
    model_override: str | None = None,
) -> Dict[str, Any]:
    if model_override:
        try:
            return client.structured_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name=schema_name,
                model_override=model_override,
            )
        except TypeError:
            # Compatibility path for test fakes and legacy client signatures.
            return client.structured_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name=schema_name,
            )
    return client.structured_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=json_schema,
        schema_name=schema_name,
    )


def extract_snapshot(
    *,
    client: OpenAIResponsesClient,
    chunks: List[TextChunk],
    tracked_fields: Iterable[str],
    assessment_context: str | None = None,
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
        max_chunks=max(1, _env_int("HR_REPORT_SNAPSHOT_MAX_CHUNKS", 120)),
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
    max_prompt_chars = max(1, _env_int("HR_REPORT_SNAPSHOT_MAX_PROMPT_CHARS", 75_000))
    estimated_prompt_chars = len(
        build_snapshot_user_prompt(
            chunks=prepared_chunks,
            tracked_fields=tracked_fields_list,
            assessment_context=assessment_context,
            prompt_overrides=prompt_overrides,
        )
    )
    if estimated_prompt_chars > max_prompt_chars and len(prepared_chunks) > 1:
        batch_size = max(1, _env_int("HR_REPORT_SNAPSHOT_CHUNK_BATCH_SIZE", 36))
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
                assessment_context=assessment_context,
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
            assessment_context=assessment_context,
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
    snapshot = _apply_exact_headcount_from_chunks(
        snapshot=snapshot,
        chunks=chunks,
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


def extract_snapshot_from_evidence(
    *,
    client: OpenAIResponsesClient,
    field_evidence: Dict[str, List[Dict[str, Any]]],
    tracked_fields: Iterable[str],
    retrieval_statuses: Dict[str, str] | None = None,
    retrieval_queries: Dict[str, List[str]] | None = None,
    prompt_overrides: Dict[str, Any] | None = None,
    company_context: str | None = None,
    assessment_context: str | None = None,
    source_chunks: List[TextChunk] | None = None,
    analysis_capture: Dict[str, Any] | None = None,
) -> CompanyPeopleSnapshot:
    tracked_fields_list = list(tracked_fields)
    retrieval_statuses = retrieval_statuses or {}
    retrieval_queries = retrieval_queries or {}

    evidence_payload: Dict[str, List[Dict[str, Any]]] = {}
    for field_path in tracked_fields_list:
        rows = field_evidence.get(field_path, []) if isinstance(field_evidence, dict) else []
        sanitized_rows: List[Dict[str, Any]] = []
        for row in rows[:12]:
            if not isinstance(row, dict):
                continue
            snippet = str(row.get("snippet", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()
            if not snippet or not chunk_id:
                continue
            sanitized_rows.append(
                {
                    "chunk_id": chunk_id,
                    "snippet": snippet,
                    "source_id": row.get("source_id"),
                    "doc_id": row.get("doc_id"),
                    "start_char": row.get("start_char"),
                    "end_char": row.get("end_char"),
                    "retrieval_score": row.get("retrieval_score"),
                    "kind": row.get("kind"),
                }
            )
        evidence_payload[field_path] = sanitized_rows

    prompt_evidence_payload = _compact_evidence_payload_for_prompt(evidence_payload)
    schema = _snapshot_schema_for_openai()
    analysis_payload: Dict[str, Any] | None = None
    resolver_model = os.getenv("OPENAI_RESOLVER_MODEL", "gpt-5-mini").strip() or None

    def _run_analysis_for_fields(fields: List[str]) -> Dict[str, Any]:
        compact_rows = {field: prompt_evidence_payload.get(field, []) for field in fields}
        return _call_structured_json(
            client,
            system_prompt=SNAPSHOT_ANALYST_SYSTEM_PROMPT,
            user_prompt=build_snapshot_analysis_prompt(
                tracked_fields=fields,
                evidence_by_field=compact_rows,
                retrieval_statuses={field: retrieval_statuses.get(field, "NOT_RETRIEVED") for field in fields},
                retrieval_queries={field: retrieval_queries.get(field, []) for field in fields},
                company_context=company_context,
                assessment_context=assessment_context,
                prompt_overrides=prompt_overrides,
            ),
            json_schema=_snapshot_analysis_schema(),
            schema_name="snapshot_field_analysis",
        )

    def _compact_rows_for_resolver(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        for row in rows:
            field_path = str(row.get("field_path", "")).strip()
            if not field_path:
                continue
            citations: List[Dict[str, str]] = []
            for citation in row.get("citations", [])[:2]:
                if not isinstance(citation, dict):
                    continue
                chunk_id = str(citation.get("chunk_id", "")).strip()
                snippet = " ".join(str(citation.get("snippet", "")).split()).strip()
                if not chunk_id or not snippet:
                    continue
                if len(snippet) > 280:
                    snippet = f"{snippet[:280].rstrip()}..."
                citations.append({"chunk_id": chunk_id, "snippet": snippet})
            value_text = " ".join(str(row.get("value_text", "")).split()).strip()
            reasoning = " ".join(str(row.get("hr_reasoning", "")).split()).strip()
            if len(value_text) > 120:
                value_text = f"{value_text[:120].rstrip()}..."
            if len(reasoning) > 240:
                reasoning = f"{reasoning[:240].rstrip()}..."
            compact.append(
                {
                    "field_path": field_path,
                    "verdict": str(row.get("verdict", "not_found")).strip(),
                    "support_strength": str(row.get("support_strength", "none")).strip(),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "value_text": value_text,
                    "hr_reasoning": reasoning,
                    "needs_confirmation": bool(row.get("needs_confirmation", False)),
                    "citations": citations,
                    "retrieval_status": row.get("retrieval_status"),
                }
            )
        return compact

    def _resolve_once(fields: List[str], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        resolver_prompt = build_snapshot_resolver_prompt(
            tracked_fields=fields,
            field_assessments=rows,
            company_context=company_context,
            assessment_context=assessment_context,
            prompt_overrides=prompt_overrides,
        )
        if resolver_model:
            try:
                return _call_structured_json(
                    client,
                    system_prompt=SNAPSHOT_RESOLVER_SYSTEM_PROMPT,
                    user_prompt=resolver_prompt,
                    json_schema=schema,
                    schema_name="company_people_snapshot",
                    model_override=resolver_model,
                )
            except Exception:
                logger.exception("snapshot_resolver_with_light_model_failed model=%s", resolver_model)
        return _call_structured_json(
            client,
            system_prompt=SNAPSHOT_RESOLVER_SYSTEM_PROMPT,
            user_prompt=resolver_prompt,
            json_schema=schema,
            schema_name="company_people_snapshot",
        )

    def _resolve_snapshot_batch(fields: List[str], analysis_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows_by_field = {
            str(row.get("field_path", "")).strip(): row
            for row in _compact_rows_for_resolver(analysis_rows)
            if str(row.get("field_path", "")).strip()
        }
        batch_rows = [rows_by_field[field] for field in fields if field in rows_by_field]
        return _resolve_once(fields, batch_rows)

    if client.is_enabled():
        try:
            field_batch_size = max(1, _env_int("HR_REPORT_SNAPSHOT_EVIDENCE_FIELD_BATCH_SIZE", 12))
            field_batches = _field_batches(tracked_fields_list, field_batch_size)
            analysis_payloads: List[Dict[str, Any]] = []
            resolver_payloads: List[Dict[str, Any]] = []
            logger.info(
                "snapshot_two_pass_batches tracked_fields=%s batches=%s batch_size=%s",
                len(tracked_fields_list),
                len(field_batches),
                field_batch_size,
            )
            for index, field_batch in enumerate(field_batches, start=1):
                batch_raw_evidence = {
                    field: evidence_payload.get(field, []) for field in field_batch
                }
                logger.info(
                    "snapshot_analysis_batch_start batch=%s/%s fields=%s",
                    index,
                    len(field_batches),
                    len(field_batch),
                )
                try:
                    batch_analysis_payload = _run_analysis_for_fields(field_batch)
                except Exception:
                    logger.exception(
                        "snapshot_analysis_batch_failed batch=%s/%s fallback=deterministic",
                        index,
                        len(field_batches),
                    )
                    batch_analysis_payload = _deterministic_analysis_payload_from_evidence(
                        evidence_payload=batch_raw_evidence,
                        tracked_fields=field_batch,
                        retrieval_statuses={field: retrieval_statuses.get(field, "NOT_RETRIEVED") for field in field_batch},
                        retrieval_queries={field: retrieval_queries.get(field, []) for field in field_batch},
                    )
                analysis_payloads.append(batch_analysis_payload)
                logger.info("snapshot_analysis_batch_done batch=%s/%s", index, len(field_batches))

                batch_analysis_rows = _analysis_rows_from_payload(batch_analysis_payload, field_batch)
                logger.info(
                    "snapshot_resolver_batch_start batch=%s/%s fields=%s",
                    index,
                    len(field_batches),
                    len(field_batch),
                )
                try:
                    batch_payload = _resolve_snapshot_batch(field_batch, batch_analysis_rows)
                except Exception:
                    logger.exception(
                        "snapshot_resolver_batch_failed batch=%s/%s fallback=deterministic",
                        index,
                        len(field_batches),
                    )
                    batch_payload = _snapshot_payload_from_analysis(
                        analysis_rows=batch_analysis_rows,
                        evidence_payload=batch_raw_evidence,
                        tracked_fields=field_batch,
                    )
                resolver_payloads.append(batch_payload)
                logger.info("snapshot_resolver_batch_done batch=%s/%s", index, len(field_batches))

            analysis_payload = _merge_analysis_payloads(
                payloads=analysis_payloads,
                tracked_fields=tracked_fields_list,
            )
            payload = _merge_snapshot_payloads(
                payloads=resolver_payloads,
                tracked_fields=tracked_fields_list,
            )
            logger.info(
                "snapshot_resolver_done fields=%s resolver_model=%s batches=%s",
                len(tracked_fields_list),
                resolver_model or client.model,
                len(field_batches),
            )
        except Exception:  # pragma: no cover - resilient fallback
            logger.exception("snapshot_two_pass_failed_fallback_to_deterministic")
            analysis_payload = _deterministic_analysis_payload_from_evidence(
                evidence_payload=evidence_payload,
                tracked_fields=tracked_fields_list,
                retrieval_statuses=retrieval_statuses,
                retrieval_queries=retrieval_queries,
            )
            payload = _snapshot_payload_from_analysis(
                analysis_rows=_analysis_rows_from_payload(analysis_payload, tracked_fields_list),
                evidence_payload=evidence_payload,
                tracked_fields=tracked_fields_list,
            )
    else:
        field_batch_size = max(1, _env_int("HR_REPORT_SNAPSHOT_EVIDENCE_FIELD_BATCH_SIZE", 12))
        field_batches = _field_batches(tracked_fields_list, field_batch_size)
        analysis_payloads: List[Dict[str, Any]] = []
        resolver_payloads: List[Dict[str, Any]] = []
        for field_batch in field_batches:
            batch_raw_evidence = {field: evidence_payload.get(field, []) for field in field_batch}
            batch_analysis_payload = _deterministic_analysis_payload_from_evidence(
                evidence_payload=batch_raw_evidence,
                tracked_fields=field_batch,
                retrieval_statuses={field: retrieval_statuses.get(field, "NOT_RETRIEVED") for field in field_batch},
                retrieval_queries={field: retrieval_queries.get(field, []) for field in field_batch},
            )
            analysis_payloads.append(batch_analysis_payload)
            resolver_payloads.append(
                _snapshot_payload_from_analysis(
                    analysis_rows=_analysis_rows_from_payload(batch_analysis_payload, field_batch),
                    evidence_payload=batch_raw_evidence,
                    tracked_fields=field_batch,
                )
            )
        analysis_payload = _merge_analysis_payloads(
            payloads=analysis_payloads,
            tracked_fields=tracked_fields_list,
        )
        payload = _merge_snapshot_payloads(
            payloads=resolver_payloads,
            tracked_fields=tracked_fields_list,
        )

    if analysis_capture is not None:
        analysis_capture.clear()
        analysis_capture.update(
            {
                "field_assessments": _analysis_rows_from_payload(analysis_payload, tracked_fields_list),
                "resolver_model": resolver_model or client.model,
                "analysis_mode": "two_pass_strict_json",
                "tracked_fields": list(tracked_fields_list),
            }
        )

    payload = _coerce_evidence_map_entries_to_dict(payload)

    # Apply analyst judgments over payload statuses and values for conservative consistency.
    analysis_rows = _analysis_rows_from_payload(analysis_payload, tracked_fields_list)
    if analysis_rows:
        payload = _overlay_analysis_on_snapshot_payload(
            payload=payload,
            analysis_rows=analysis_rows,
            evidence_payload=evidence_payload,
        )

    # Anchor model citations using richer metadata from evidence payload.
    citation_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for rows in evidence_payload.values():
        for row in rows:
            key = (str(row.get("chunk_id", "")).strip(), str(row.get("snippet", "")).strip())
            if key[0] and key[1]:
                citation_lookup[key] = row

    evidence_map = payload.get("evidence_map", {})
    if isinstance(evidence_map, dict):
        for field_path, evidence in evidence_map.items():
            if not isinstance(evidence, dict):
                continue
            anchored: List[Dict[str, Any]] = []
            for citation in evidence.get("citations", []):
                if not isinstance(citation, dict):
                    continue
                chunk_id = str(citation.get("chunk_id", "")).strip()
                snippet = str(citation.get("snippet", "")).strip()
                if not chunk_id or not snippet:
                    continue
                lookup = citation_lookup.get((chunk_id, snippet))
                merged = {"chunk_id": chunk_id, "snippet": snippet, "evidence_recovered_by": "llm"}
                if lookup is not None:
                    for key in ["source_id", "doc_id", "start_char", "end_char", "retrieval_score"]:
                        if lookup.get(key) is not None:
                            merged[key] = lookup.get(key)
                    if merged.get("evidence_recovered_by") is None:
                        merged["evidence_recovered_by"] = "targeted_retrieval"
                anchored.append(merged)
            evidence["citations"] = anchored
            evidence["match_count"] = len(anchored)
            evidence_map[field_path] = evidence
        payload["evidence_map"] = evidence_map

    snapshot = CompanyPeopleSnapshot.model_validate(payload)

    # Deterministic supplements run over evidence snippets (converted into pseudo-chunks).
    pseudo_chunks: List[TextChunk] = []
    for field_path, rows in evidence_payload.items():
        for row in rows:
            snippet = str(row.get("snippet", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()
            if not snippet or not chunk_id:
                continue
            pseudo_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=str(row.get("doc_id") or "doc-evidence"),
                    section=field_path,
                    text=snippet,
                    start_char=(int(row["start_char"]) if row.get("start_char") is not None else None),
                    end_char=(int(row["end_char"]) if row.get("end_char") is not None else None),
                )
            )
    snapshot = _apply_explicit_chunk_supplements(
        snapshot=snapshot,
        chunks=pseudo_chunks,
        tracked_fields=tracked_fields_list,
    )

    # Retrieval overlay for deterministic downstream checks.
    for field_path in tracked_fields_list:
        evidence = snapshot.evidence_map.get(field_path)
        if evidence is None:
            evidence = EvidenceStatus(status="not_provided_in_sources", citations=[])
            snapshot.evidence_map[field_path] = evidence

        status_guess = retrieval_statuses.get(field_path)
        evidence.retrieval_status = status_guess
        evidence.retrieval_queries = list(retrieval_queries.get(field_path, []))[:12]
        rows = evidence_payload.get(field_path, [])
        evidence.match_count = len(rows)

        if status_guess == "MENTIONED_EXPLICIT" and evidence.status != "present" and rows:
            top = rows[0]
            citation = Citation(
                chunk_id=str(top.get("chunk_id")),
                snippet=str(top.get("snippet")),
                source_id=(str(top.get("source_id")) if top.get("source_id") else None),
                doc_id=(str(top.get("doc_id")) if top.get("doc_id") else None),
                start_char=(int(top["start_char"]) if top.get("start_char") is not None else None),
                end_char=(int(top["end_char"]) if top.get("end_char") is not None else None),
                evidence_recovered_by="targeted_retrieval",
                retrieval_score=(float(top["retrieval_score"]) if top.get("retrieval_score") is not None else None),
            )
            evidence.status = "present"
            evidence.citations = [citation]
            evidence.match_count = max(1, len(rows))

        if status_guess in {"MENTIONED_IMPLICIT", "MENTIONED_AMBIGUOUS"} and evidence.status != "explicitly_missing":
            if evidence.status != "present":
                evidence.status = "not_provided_in_sources"
            if rows:
                top = rows[0]
                evidence.citations = [
                    Citation(
                        chunk_id=str(top.get("chunk_id")),
                        snippet=str(top.get("snippet")),
                        source_id=(str(top.get("source_id")) if top.get("source_id") else None),
                        doc_id=(str(top.get("doc_id")) if top.get("doc_id") else None),
                        start_char=(int(top["start_char"]) if top.get("start_char") is not None else None),
                        end_char=(int(top["end_char"]) if top.get("end_char") is not None else None),
                        evidence_recovered_by="targeted_retrieval",
                        retrieval_score=(
                            float(top["retrieval_score"]) if top.get("retrieval_score") is not None else None
                        ),
                    )
                ]

        if status_guess == "NOT_RETRIEVED" and evidence.status not in {"present", "explicitly_missing"}:
            evidence.status = "not_provided_in_sources"
            evidence.citations = []

        if status_guess == "NOT_FOUND_IN_RETRIEVED" and evidence.status not in {"present", "explicitly_missing"}:
            evidence.status = "not_provided_in_sources"
            if not evidence.citations:
                evidence.citations = []

    snapshot = _apply_exact_headcount_from_chunks(
        snapshot=snapshot,
        chunks=(source_chunks if source_chunks else pseudo_chunks),
        tracked_fields=tracked_fields_list,
    )

    return snapshot


def _extract_exact_headcount_from_sentence(sentence: str) -> tuple[int, int] | None:
    lower = sentence.lower()
    if HEADCOUNT_RANGE_OR_PLUS_RE.search(lower):
        return None
    if HEADCOUNT_INEXACT_RE.search(lower):
        return None
    for pattern, base_score in HEADCOUNT_EXACT_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        raw_value = next((group for group in match.groups() if group and group.isdigit()), None)
        if raw_value is None:
            continue
        value = int(raw_value)
        if value <= 0:
            continue
        score = base_score
        if "headcount" in lower:
            score += 2
        if any(token in lower for token in ["employees", "employee", "fte", "team members"]):
            score += 1
        return value, score
    return None


def _best_exact_headcount_candidate(chunks: List[TextChunk]) -> tuple[int, Dict[str, Any]] | None:
    if not chunks:
        return None
    counts: Dict[int, int] = {}
    best_by_value: Dict[int, tuple[int, Dict[str, Any]]] = {}

    for chunk in chunks:
        sentences = _split_sentences(chunk.text)
        if not sentences:
            sentences = [" ".join(chunk.text.split()).strip()]
        for sentence in sentences:
            normalized = " ".join(sentence.split()).strip()
            if not normalized:
                continue
            parsed = _extract_exact_headcount_from_sentence(normalized)
            if parsed is None:
                continue
            value, score = parsed
            counts[value] = counts.get(value, 0) + 1
            citation = {
                "chunk_id": chunk.chunk_id,
                "snippet": normalized,
                "doc_id": chunk.doc_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "evidence_recovered_by": "deterministic",
            }
            current = best_by_value.get(value)
            if current is None or score > current[0]:
                best_by_value[value] = (score, citation)

    if not best_by_value:
        return None
    selected_value = max(best_by_value.keys(), key=lambda value: (counts.get(value, 0), best_by_value[value][0]))
    return selected_value, best_by_value[selected_value][1]


def _dedupe_citation_rows(citations: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    if not isinstance(citations, list):
        return out
    for item in citations:
        if isinstance(item, Citation):
            item = item.model_dump(mode="python")
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        if not chunk_id or not snippet:
            continue
        key = (chunk_id, snippet)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _apply_exact_headcount_from_chunks(
    *,
    snapshot: CompanyPeopleSnapshot,
    chunks: List[TextChunk],
    tracked_fields: List[str],
) -> CompanyPeopleSnapshot:
    if not chunks:
        return snapshot
    if tracked_fields and "headcount" not in tracked_fields:
        return snapshot

    candidate = _best_exact_headcount_candidate(chunks)
    if candidate is None:
        return snapshot
    headcount_value, citation = candidate

    payload = snapshot.model_dump(mode="python")
    payload["headcount"] = headcount_value

    evidence_map = dict(payload.get("evidence_map", {}))
    existing = evidence_map.get("headcount", {})
    if not isinstance(existing, dict):
        existing = {}

    existing_citations = _dedupe_citation_rows(existing.get("citations", []))
    citations = _dedupe_citation_rows([citation] + existing_citations)
    existing["status"] = "present"
    existing["retrieval_status"] = "MENTIONED_EXPLICIT"
    existing["coverage_note"] = existing.get("coverage_note")
    existing["retrieval_queries"] = list(existing.get("retrieval_queries", []))
    existing["match_count"] = max(1, int(existing.get("match_count", 0)))
    existing["citations"] = citations

    evidence_map["headcount"] = existing
    payload["evidence_map"] = evidence_map
    return CompanyPeopleSnapshot.model_validate(payload)


def _compact_evidence_payload_for_prompt(
    evidence_payload: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    compact: Dict[str, List[Dict[str, Any]]] = {}
    for field_path, rows in evidence_payload.items():
        compact_rows: List[Dict[str, Any]] = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            chunk_id = str(row.get("chunk_id", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            if not chunk_id or not snippet:
                continue
            dedupe_key = (chunk_id, snippet)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            prompt_row: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "snippet": snippet,
            }
            kind = row.get("kind")
            if kind:
                prompt_row["kind"] = str(kind)
            compact_rows.append(prompt_row)
        compact[field_path] = compact_rows
    return compact


def _deterministic_snapshot_payload_from_evidence(
    *,
    evidence_payload: Dict[str, List[Dict[str, Any]]],
    tracked_fields: List[str],
) -> Dict[str, Any]:
    payload = CompanyPeopleSnapshot().model_dump(mode="python")
    evidence_rows: List[Dict[str, Any]] = []

    for field_path in tracked_fields:
        rows = evidence_payload.get(field_path, [])
        snippets = [str(row.get("snippet", "")).strip() for row in rows if isinstance(row, dict)]
        joined = " ".join(snippets).lower()
        citations = [
            {"chunk_id": str(row.get("chunk_id", "")).strip(), "snippet": str(row.get("snippet", "")).strip()}
            for row in rows
            if str(row.get("chunk_id", "")).strip() and str(row.get("snippet", "")).strip()
        ]
        status = "not_provided_in_sources"

        if citations and any(re.search(pattern, joined) for pattern in NEGATIVE_PATTERNS):
            status = "explicitly_missing"
        elif citations:
            status = "present"

        if field_path in BOOLEAN_TRACKED_FIELDS and citations:
            _set_path(payload, field_path, status == "present")
        elif field_path == "hris_data.hris_system":
            for system in HRIS_SYSTEM_NAMES:
                if re.search(rf"\b{re.escape(system.lower())}\b", joined):
                    _set_path(payload, field_path, system)
                    status = "present"
                    break
        elif field_path == "headcount":
            match = re.search(r"\b(\d{2,5})\s*(employees|employee|fte|team members)\b", joined)
            if match:
                _set_path(payload, field_path, int(match.group(1)))
                status = "present"
        elif field_path == "headcount_range":
            match = re.search(r"\b(\d{2,5}\s*(?:-|to)\s*\d{2,5}|\d{2,5}\+)\b", joined)
            if match:
                _set_path(payload, field_path, match.group(1))
                status = "present"

        evidence_rows.append(
            {
                "field_path": field_path,
                "status": status,
                "citations": citations,
            }
        )

    payload["evidence_map"] = evidence_rows
    return payload


def _snapshot_analysis_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "field_assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "field_path": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "present",
                                "explicitly_missing",
                                "ambiguous",
                                "conflict",
                                "not_found",
                            ],
                        },
                        "support_strength": {
                            "type": "string",
                            "enum": ["strong", "moderate", "weak", "none"],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "value_text": {"type": "string"},
                        "hr_reasoning": {"type": "string"},
                        "needs_confirmation": {"type": "boolean"},
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
                    "required": [
                        "field_path",
                        "verdict",
                        "support_strength",
                        "confidence",
                        "value_text",
                        "hr_reasoning",
                        "needs_confirmation",
                        "citations",
                    ],
                },
            }
        },
        "required": ["field_assessments"],
    }


def _analysis_rank(verdict: str, confidence: float, citations: int) -> tuple[int, float, int]:
    verdict_rank = {
        "present": 5,
        "explicitly_missing": 5,
        "conflict": 4,
        "ambiguous": 3,
        "not_found": 2,
    }.get(verdict, 1)
    return (verdict_rank, float(confidence), int(citations))


def _extract_headcount_values_from_rows(rows: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for row in rows:
        snippet = str(row.get("snippet", "")).strip()
        if not snippet:
            continue
        lower = snippet.lower()
        if HEADCOUNT_RANGE_OR_PLUS_RE.search(lower) or HEADCOUNT_INEXACT_RE.search(lower):
            continue
        for pattern, _ in HEADCOUNT_EXACT_PATTERNS:
            match = pattern.search(snippet)
            if not match:
                continue
            raw = next((group for group in match.groups() if group and group.isdigit()), None)
            if raw and raw.isdigit():
                out.append(int(raw))
    return out


def _best_value_text_for_field(field_path: str, rows: List[Dict[str, Any]]) -> str:
    snippets = [str(row.get("snippet", "")).strip() for row in rows if isinstance(row, dict)]
    joined = " ".join(snippets)
    lower = joined.lower()

    if field_path == "headcount":
        values = _extract_headcount_values_from_rows(rows)
        if values:
            return str(max(set(values), key=values.count))
        return ""
    if field_path == "headcount_range":
        match = re.search(r"\b(\d{1,5}\s*(?:-|to)\s*\d{1,5}|\d{1,5}\+)\b", joined, flags=re.IGNORECASE)
        return match.group(1) if match else ""
    if field_path == "hris_data.hris_system":
        for system in HRIS_SYSTEM_NAMES:
            if re.search(rf"\b{re.escape(system.lower())}\b", lower):
                return system
        return ""
    if field_path in BOOLEAN_TRACKED_FIELDS:
        if any(re.search(pattern, lower) for pattern in NEGATIVE_PATTERNS):
            return "false"
        if any(re.search(pattern, lower) for pattern in POSITIVE_PATTERNS):
            return "true"
    return ""


def _deterministic_analysis_payload_from_evidence(
    *,
    evidence_payload: Dict[str, List[Dict[str, Any]]],
    tracked_fields: List[str],
    retrieval_statuses: Dict[str, str],
    retrieval_queries: Dict[str, List[str]],
) -> Dict[str, Any]:
    rows_out: List[Dict[str, Any]] = []
    for field_path in tracked_fields:
        rows = evidence_payload.get(field_path, []) if isinstance(evidence_payload, dict) else []
        citations = [
            {
                "chunk_id": str(row.get("chunk_id", "")).strip(),
                "snippet": str(row.get("snippet", "")).strip(),
            }
            for row in rows
            if str(row.get("chunk_id", "")).strip() and str(row.get("snippet", "")).strip()
        ][:3]
        joined = " ".join(citation["snippet"] for citation in citations).lower()
        retrieval_status = str(retrieval_statuses.get(field_path, "NOT_RETRIEVED"))
        value_text = _best_value_text_for_field(field_path, rows)
        headcount_values = _extract_headcount_values_from_rows(rows) if field_path == "headcount" else []
        if len(set(headcount_values)) > 1:
            verdict = "conflict"
            support_strength = "weak"
            confidence = 0.52
            value_text = ""
            reasoning = "Multiple conflicting explicit headcount values were found."
        elif not citations:
            verdict = "not_found"
            support_strength = "none"
            confidence = 0.14 if retrieval_status == "NOT_RETRIEVED" else 0.22
            reasoning = "No reliable field-specific evidence was found in retrieved snippets."
        elif any(re.search(pattern, joined) for pattern in NEGATIVE_PATTERNS):
            verdict = "explicitly_missing"
            support_strength = "strong" if retrieval_status == "MENTIONED_EXPLICIT" else "moderate"
            confidence = 0.84 if retrieval_status == "MENTIONED_EXPLICIT" else 0.74
            reasoning = "Evidence explicitly indicates absence or missing control."
        elif retrieval_status == "MENTIONED_EXPLICIT" or any(
            str(row.get("kind", "")).strip().lower() in {"explicit", "explicit_absence"} for row in rows
        ):
            verdict = "present"
            support_strength = "strong"
            confidence = 0.86
            reasoning = "Explicit HR evidence indicates the field is present."
        elif retrieval_status in {"MENTIONED_IMPLICIT", "MENTIONED_AMBIGUOUS"}:
            verdict = "ambiguous"
            support_strength = "weak"
            confidence = 0.44 if retrieval_status == "MENTIONED_IMPLICIT" else 0.34
            reasoning = "Signals exist but evidence is indirect or weak for definitive confirmation."
        else:
            verdict = "not_found"
            support_strength = "none"
            confidence = 0.2
            reasoning = "Evidence does not clearly support this field."

        rows_out.append(
            {
                "field_path": field_path,
                "verdict": verdict,
                "support_strength": support_strength,
                "confidence": round(float(confidence), 3),
                "value_text": value_text,
                "hr_reasoning": reasoning,
                "needs_confirmation": verdict in {"ambiguous", "conflict", "not_found"},
                "citations": citations,
                "retrieval_status": retrieval_status,
                "retrieval_queries": list(retrieval_queries.get(field_path, []))[:12],
            }
        )
    return {"field_assessments": rows_out}


def _analysis_rows_from_payload(payload: Dict[str, Any] | None, tracked_fields: List[str]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        payload = {}
    rows = payload.get("field_assessments", [])
    by_field: Dict[str, Dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            field_path = str(row.get("field_path", "")).strip()
            if not field_path:
                continue
            normalized = {
                "field_path": field_path,
                "verdict": str(row.get("verdict", "not_found")),
                "support_strength": str(row.get("support_strength", "none")),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "value_text": str(row.get("value_text", "") or ""),
                "hr_reasoning": str(row.get("hr_reasoning", "") or ""),
                "needs_confirmation": bool(row.get("needs_confirmation", False)),
                "citations": _dedupe_citations(row.get("citations", [])),
                "retrieval_status": row.get("retrieval_status"),
                "retrieval_queries": row.get("retrieval_queries", []),
            }
            current = by_field.get(field_path)
            if current is None or _analysis_rank(
                normalized["verdict"], normalized["confidence"], len(normalized["citations"])
            ) > _analysis_rank(current["verdict"], current["confidence"], len(current["citations"])):
                by_field[field_path] = normalized

    out: List[Dict[str, Any]] = []
    for field_path in tracked_fields:
        row = by_field.get(field_path)
        if row is None:
            row = {
                "field_path": field_path,
                "verdict": "not_found",
                "support_strength": "none",
                "confidence": 0.0,
                "value_text": "",
                "hr_reasoning": "",
                "needs_confirmation": True,
                "citations": [],
                "retrieval_status": None,
                "retrieval_queries": [],
            }
        out.append(row)
    return out


def _merge_analysis_payloads(
    *,
    payloads: List[Dict[str, Any]],
    tracked_fields: List[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        rows.extend(_analysis_rows_from_payload(payload, tracked_fields))
    return {"field_assessments": _analysis_rows_from_payload({"field_assessments": rows}, tracked_fields)}


def _parse_value_text_for_field(field_path: str, value_text: str) -> Any:
    raw = (value_text or "").strip()
    if not raw:
        return None
    lower = raw.lower()
    if field_path == "headcount":
        match = re.search(r"\b(\d{1,5})\b", raw)
        return int(match.group(1)) if match else None
    if field_path == "headcount_range":
        match = re.search(r"\b(\d{1,5}\s*(?:-|to)\s*\d{1,5}|\d{1,5}\+)\b", raw, flags=re.IGNORECASE)
        return match.group(1) if match else None
    if field_path in BOOLEAN_TRACKED_FIELDS:
        if lower in {"true", "yes", "present", "in place", "documented"}:
            return True
        if lower in {"false", "no", "missing", "not present"}:
            return False
        if any(re.search(pattern, lower) for pattern in NEGATIVE_PATTERNS):
            return False
        if any(re.search(pattern, lower) for pattern in POSITIVE_PATTERNS):
            return True
        return None
    if field_path == "hris_data.hris_system":
        return raw
    return None


def _status_from_verdict(verdict: str) -> str:
    if verdict == "present":
        return "present"
    if verdict == "explicitly_missing":
        return "explicitly_missing"
    return "not_provided_in_sources"


def _overlay_analysis_on_snapshot_payload(
    *,
    payload: Dict[str, Any],
    analysis_rows: List[Dict[str, Any]],
    evidence_payload: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    evidence_map = payload.get("evidence_map", {})
    if not isinstance(evidence_map, dict):
        evidence_map = {}

    for row in analysis_rows:
        field_path = str(row.get("field_path", "")).strip()
        if not field_path:
            continue
        verdict = str(row.get("verdict", "not_found")).strip()
        status = _status_from_verdict(verdict)
        citations = _dedupe_citations(row.get("citations", []))

        if not citations:
            rows = evidence_payload.get(field_path, [])
            citations = _dedupe_citations(
                [
                    {
                        "chunk_id": str(candidate.get("chunk_id", "")).strip(),
                        "snippet": str(candidate.get("snippet", "")).strip(),
                    }
                    for candidate in rows[:2]
                    if str(candidate.get("chunk_id", "")).strip() and str(candidate.get("snippet", "")).strip()
                ]
            )

        evidence = evidence_map.get(field_path, {})
        if not isinstance(evidence, dict):
            evidence = {}
        evidence["status"] = status
        evidence["citations"] = citations if status in {"present", "explicitly_missing"} else []
        evidence["match_count"] = len(citations)
        if row.get("retrieval_status"):
            evidence["retrieval_status"] = row.get("retrieval_status")
        if isinstance(row.get("retrieval_queries"), list):
            evidence["retrieval_queries"] = list(row.get("retrieval_queries", []))[:12]
        evidence_map[field_path] = evidence

        parsed_value = _parse_value_text_for_field(field_path, str(row.get("value_text", "")))
        if status == "present" and parsed_value is not None:
            _set_path(payload, field_path, parsed_value)
        elif status != "present" and field_path in {"headcount", "headcount_range"}:
            _set_path(payload, field_path, None)

    payload["evidence_map"] = evidence_map
    return payload


def _snapshot_payload_from_analysis(
    *,
    analysis_rows: List[Dict[str, Any]],
    evidence_payload: Dict[str, List[Dict[str, Any]]],
    tracked_fields: List[str],
) -> Dict[str, Any]:
    payload = _deterministic_snapshot_payload_from_evidence(
        evidence_payload=evidence_payload,
        tracked_fields=tracked_fields,
    )
    payload = _coerce_evidence_map_entries_to_dict(payload)
    return _overlay_analysis_on_snapshot_payload(
        payload=payload,
        analysis_rows=analysis_rows,
        evidence_payload=evidence_payload,
    )


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


def _field_batches(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_snapshot_payload_with_retry(
    *,
    client: OpenAIResponsesClient,
    chunks: List[TextChunk],
    tracked_fields: List[str],
    assessment_context: str | None,
    prompt_overrides: Dict[str, Any] | None,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        return _extract_snapshot_payload_once(
            client=client,
            chunks=chunks,
            tracked_fields=tracked_fields,
            assessment_context=assessment_context,
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
            assessment_context=assessment_context,
            prompt_overrides=prompt_overrides,
            schema=schema,
        )
        right = _extract_snapshot_payload_with_retry(
            client=client,
            chunks=chunks[midpoint:],
            tracked_fields=tracked_fields,
            assessment_context=assessment_context,
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
    assessment_context: str | None,
    prompt_overrides: Dict[str, Any] | None,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    return client.structured_json(
        system_prompt=SNAPSHOT_SYSTEM_PROMPT,
        user_prompt=build_snapshot_user_prompt(
            chunks=chunks,
            tracked_fields=tracked_fields,
            assessment_context=assessment_context,
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
