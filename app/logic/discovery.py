from __future__ import annotations

import os
import re
from typing import Any, Dict, List
import logging

from app.llm.client import OpenAIResponsesClient
from app.llm.prompts import DISCOVERY_SYSTEM_PROMPT, build_discovery_user_prompt
from app.models import Citation, ConsultantProfile, Finding, Subcheck

logger = logging.getLogger(__name__)


def _sentence_window(text: str, start: int, end: int, max_chars: int = 360) -> str:
    left_candidates = [text.rfind(token, 0, start) for token in [".", "!", "?", "\n"]]
    right_candidates = [idx for token in [".", "!", "?", "\n"] if (idx := text.find(token, end)) != -1]

    left = max(left_candidates)
    right = min(right_candidates) if right_candidates else len(text) - 1
    snippet = text[(left + 1 if left >= 0 else 0) : right + 1]
    snippet = " ".join(snippet.split()).strip()
    if len(snippet) <= max_chars:
        return snippet
    return f"{snippet[:max_chars].rstrip()}..."


def _citation_from_row(row: Dict[str, Any]) -> Citation | None:
    snippet = str(row.get("snippet", "")).strip()
    chunk_id = str(row.get("chunk_id", "")).strip()
    if not snippet or not chunk_id:
        return None
    return Citation(
        chunk_id=chunk_id,
        snippet=snippet,
        source_id=(str(row.get("source_id")) if row.get("source_id") else None),
        doc_id=(str(row.get("doc_id")) if row.get("doc_id") else None),
        start_char=(int(row["start_char"]) if row.get("start_char") is not None else None),
        end_char=(int(row["end_char"]) if row.get("end_char") is not None else None),
        evidence_recovered_by=(str(row.get("evidence_recovered_by")) if row.get("evidence_recovered_by") else None),
        retrieval_score=(float(row["retrieval_score"]) if row.get("retrieval_score") is not None else None),
    )


def _red_flag_findings(profile: ConsultantProfile, evidence_result: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []
    chunk_text_map = evidence_result.get("chunk_text_map", {})
    if not isinstance(chunk_text_map, dict):
        return findings

    for rule in profile.red_flags:
        pattern = re.compile(rule.pattern, flags=re.IGNORECASE) if rule.regex else None
        matches: List[Citation] = []
        for chunk_id, text in chunk_text_map.items():
            text_value = str(text)
            hit = pattern.search(text_value) if pattern else (rule.pattern.lower() in text_value.lower())
            if not hit:
                continue
            snippet = (
                _sentence_window(text_value, hit.start(), hit.end())
                if pattern
                else _sentence_window(text_value, 0, min(len(text_value), 160))
            )
            matches.append(
                Citation(
                    chunk_id=str(chunk_id),
                    snippet=snippet,
                    evidence_recovered_by="deterministic",
                )
            )
            if len(matches) >= 3:
                break

        if not matches:
            continue

        findings.append(
            Finding(
                check_id=f"discovery:red_flag:{rule.id}",
                area=rule.area,
                title=f"Potential red flag: {rule.id.replace('_', ' ')}",
                severity=rule.severity,
                evidence_status="present",
                retrieval_status="MENTIONED_EXPLICIT",
                needs_confirmation=False,
                is_threshold_prompt=rule.severity in {"high", "critical"},
                stage_reason="Pattern-level risk signal detected in reviewed sources.",
                evidence=matches,
                subchecks=[
                    Subcheck(
                        capability_key=rule.id,
                        evidence_status="present",
                        retrieval_status="MENTIONED_EXPLICIT",
                        citations=matches,
                    )
                ],
                actions=["Review incident details and validate remediation ownership."],
                owner="TBD / assign (e.g., HR/People, Finance, Ops)",
                metrics=[],
                questions=([rule.question] if rule.question else []),
            )
        )
    return findings


def _catalog_from_evidence(evidence_result: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    expectation_evidence = evidence_result.get("expectation_evidence", {})
    if not isinstance(expectation_evidence, dict):
        return rows
    for expectation_id, evidence_rows in expectation_evidence.items():
        if not isinstance(evidence_rows, list):
            continue
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            snippet = str(row.get("snippet", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()
            if not snippet or not chunk_id:
                continue
            rows.append(
                {
                    "expectation_id": expectation_id,
                    "chunk_id": chunk_id,
                    "snippet": snippet,
                    "source": row.get("source"),
                    "retrieval_score": row.get("retrieval_score", 0.0),
                }
            )
    rows.sort(key=lambda item: float(item.get("retrieval_score") or 0.0), reverse=True)
    return rows[:limit]


def _llm_discovery_findings(
    *,
    profile: ConsultantProfile,
    stage_label: str,
    evidence_result: Dict[str, Any],
    client: OpenAIResponsesClient,
    budget: int,
    assessment_context: str | None = None,
) -> List[Finding]:
    if budget <= 0:
        return []

    evidence_catalog = _catalog_from_evidence(evidence_result)
    if not evidence_catalog:
        return []

    schema = _discovery_schema()
    max_prompt_chars = max(4_000, _env_int("HR_REPORT_DISCOVERY_MAX_PROMPT_CHARS", 16_000))
    catalog_batch_size = max(8, _env_int("HR_REPORT_DISCOVERY_CATALOG_BATCH_SIZE", 28))

    estimated_prompt_chars = len(
        build_discovery_user_prompt(
            stage_label=stage_label,
            evidence_catalog=evidence_catalog,
            max_findings=budget,
            assessment_context=assessment_context,
            prompt_overrides=profile.prompt_overrides,
        )
    )

    if estimated_prompt_chars > max_prompt_chars and len(evidence_catalog) > catalog_batch_size:
        catalog_batches = _catalog_batches(evidence_catalog, catalog_batch_size)
        per_batch_budget = max(1, min(budget, ((budget + len(catalog_batches) - 1) // len(catalog_batches)) + 1))
        findings: List[Finding] = []
        logger.info(
            "discovery_batch_mode enabled=true catalog_rows=%s batches=%s batch_size=%s estimated_prompt_chars=%s",
            len(evidence_catalog),
            len(catalog_batches),
            catalog_batch_size,
            estimated_prompt_chars,
        )
        for index, catalog_batch in enumerate(catalog_batches, start=1):
            logger.info(
                "discovery_batch_start batch=%s/%s rows=%s",
                index,
                len(catalog_batches),
                len(catalog_batch),
            )
            payload = client.structured_json(
                system_prompt=DISCOVERY_SYSTEM_PROMPT,
                user_prompt=build_discovery_user_prompt(
                    stage_label=stage_label,
                    evidence_catalog=catalog_batch,
                    max_findings=per_batch_budget,
                    assessment_context=assessment_context,
                    prompt_overrides=profile.prompt_overrides,
                ),
                json_schema=schema,
                schema_name="discovery_observations",
            )
            findings.extend(_parse_discovery_payload(payload=payload, evidence_catalog=catalog_batch))
            logger.info("discovery_batch_done batch=%s/%s", index, len(catalog_batches))
            if len(findings) >= budget * 2:
                break
        return findings[:budget]

    logger.info(
        "discovery_batch_mode enabled=false catalog_rows=%s estimated_prompt_chars=%s",
        len(evidence_catalog),
        estimated_prompt_chars,
    )
    payload = client.structured_json(
        system_prompt=DISCOVERY_SYSTEM_PROMPT,
        user_prompt=build_discovery_user_prompt(
            stage_label=stage_label,
            evidence_catalog=evidence_catalog,
            max_findings=budget,
            assessment_context=assessment_context,
            prompt_overrides=profile.prompt_overrides,
        ),
        json_schema=schema,
        schema_name="discovery_observations",
    )

    return _parse_discovery_payload(payload=payload, evidence_catalog=evidence_catalog)[:budget]


def _parse_discovery_payload(*, payload: Dict[str, Any], evidence_catalog: List[Dict[str, Any]]) -> List[Finding]:
    by_chunk: Dict[str, Dict[str, Any]] = {row["chunk_id"]: row for row in evidence_catalog}
    findings: List[Finding] = []
    for row in payload.get("observations", []):
        if not isinstance(row, dict):
            continue

        citations: List[Citation] = []
        for chunk_id in row.get("evidence_chunk_ids", []):
            raw = by_chunk.get(str(chunk_id))
            if raw is None:
                continue
            citation = _citation_from_row(raw)
            if citation is not None:
                citations.append(citation)
            if len(citations) >= 3:
                break

        hypothesis = bool(row.get("hypothesis"))
        question = str(row.get("question", "")).strip()
        evidence_status = "present" if citations else "not_provided_in_sources"
        needs_confirmation = hypothesis or evidence_status != "present"

        findings.append(
            Finding(
                check_id=f"discovery:llm:{row.get('id', 'observation')}",
                area=str(row.get("area", "retention")),
                title=str(row.get("title", "Additional observation")),
                severity=str(row.get("severity", "medium")),
                evidence_status=evidence_status,
                retrieval_status=("MENTIONED_EXPLICIT" if citations else "NOT_FOUND_IN_RETRIEVED"),
                needs_confirmation=needs_confirmation,
                is_threshold_prompt=False,
                stage_reason=str(row.get("rationale", "Potential additional risk identified.")),
                evidence=citations,
                subchecks=[
                    Subcheck(
                        capability_key=str(row.get("id", "discovery_observation")),
                        evidence_status=evidence_status,
                        retrieval_status=("MENTIONED_EXPLICIT" if citations else "NOT_FOUND_IN_RETRIEVED"),
                        citations=citations,
                        missing_reason=(
                            "Evidence was weak; treat as hypothesis and confirm."
                            if needs_confirmation and not citations
                            else None
                        ),
                    )
                ],
                actions=["Validate signal and assign owner if confirmed."],
                owner="TBD / assign (e.g., HR/People, Finance, Ops)",
                metrics=[],
                questions=([question] if question else []),
            )
        )
    return findings


def _discovery_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "area": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "rationale": {"type": "string"},
                        "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
                        "hypothesis": {"type": "boolean"},
                        "question": {"type": "string"},
                    },
                    "required": [
                        "id",
                        "title",
                        "area",
                        "severity",
                        "rationale",
                        "evidence_chunk_ids",
                        "hypothesis",
                        "question",
                    ],
                },
            }
        },
        "required": ["observations"],
    }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _catalog_batches(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def discover_additional_observations(
    *,
    profile: ConsultantProfile,
    stage_label: str,
    evidence_result: Dict[str, Any],
    client: OpenAIResponsesClient | None = None,
    assessment_context: str | None = None,
) -> List[Finding]:
    findings: List[Finding] = _red_flag_findings(profile, evidence_result)

    if (
        profile.discovery.enabled
        and client is not None
        and client.is_enabled()
        and len(findings) < profile.discovery.max_findings
    ):
        try:
            findings.extend(
                _llm_discovery_findings(
                    profile=profile,
                    stage_label=stage_label,
                    evidence_result=evidence_result,
                    client=client,
                    budget=max(0, profile.discovery.max_findings - len(findings)),
                    assessment_context=assessment_context,
                )
            )
        except Exception:  # pragma: no cover - resilient fallback
            logger.exception("discovery_agent_failed")

    deduped: List[Finding] = []
    seen = set()
    for finding in findings:
        key = (finding.area, finding.title.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)

    return deduped[: profile.discovery.max_findings]
