from __future__ import annotations

import re
from typing import Dict, List, Tuple

from app.logic.evidence_semantics import evidence_status_requires_confirmation, map_verdict_to_evidence_status
from app.models import Citation, CompanyPeopleSnapshot, ConsultantProfile, Finding, RiskItem, Subcheck

ABSENCE_PATTERNS = [
    r"\b(no|none|missing|lacks?|lacking|absence of)\b.{0,40}\b(policy|process|program|training|framework|system|control|controls)\b",
    r"\bdoes\s+not\s+(exist|have|include|provide|maintain)\b",
    r"\bnot\s+(in place|defined|documented|established|tracked|available)\b",
    r"\bwithout\b.{0,40}\b(policy|process|program|training|framework|system|control|controls)\b",
]
SEVERITY_ORDER = ["low", "medium", "high", "critical"]
RELEVANCE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "have",
    "has",
    "are",
    "was",
    "were",
    "not",
    "can",
    "will",
    "policy",
    "process",
    "program",
    "control",
    "controls",
}


def _severity_down_one(severity: str) -> str:
    idx = max(0, SEVERITY_ORDER.index(severity) - 1)
    return SEVERITY_ORDER[idx]


def _owner_from_citations(citations: List[Citation]) -> str:
    text = " ".join(c.snippet.lower() for c in citations)
    if re.search(r"\bpeople ops\b|\bpeople team\b", text):
        return "People Team"
    if re.search(r"\bhr\b|\bhuman resources\b", text):
        return "HR Team"
    if re.search(r"\bfinance\b", text):
        return "Finance"
    if re.search(r"\boperations\b|\bops\b", text):
        return "Ops"
    return "TBD / assign (e.g., HR/People, Finance, Ops)"


def _to_citations_with_absence(rows: List[Dict[str, object]]) -> Tuple[List[Citation], List[Citation]]:
    citations: List[Citation] = []
    absence: List[Citation] = []
    for row in rows:
        snippet = str(row.get("snippet", "")).strip()
        chunk_id = str(row.get("chunk_id", "")).strip()
        if not snippet or not chunk_id:
            continue

        citation = Citation(
            chunk_id=chunk_id,
            snippet=snippet,
            source_id=(str(row.get("source_id")) if row.get("source_id") else None),
            doc_id=(str(row.get("doc_id")) if row.get("doc_id") else None),
            start_char=(int(row["start_char"]) if row.get("start_char") is not None else None),
            end_char=(int(row["end_char"]) if row.get("end_char") is not None else None),
            evidence_recovered_by=(
                str(row.get("evidence_recovered_by")) if row.get("evidence_recovered_by") else None
            ),
            retrieval_score=(float(row["retrieval_score"]) if row.get("retrieval_score") is not None else None),
        )
        citations.append(citation)

        kind = str(row.get("kind", "")).strip().lower()
        if kind == "explicit_absence":
            absence.append(citation)
            continue
        if not kind and any(re.search(pattern, snippet.lower()) for pattern in ABSENCE_PATTERNS):
            absence.append(citation)
    return citations, absence


def _expectation_lookup(profile: ConsultantProfile) -> Dict[str, object]:
    return {expectation.id: expectation for expectation in profile.expectations}


def _active_expectations(profile: ConsultantProfile, stage_id: str) -> List[object]:
    stage = next((item for item in profile.stages if item.id == stage_id), None)
    if stage is None:
        return list(profile.expectations)
    expectation_map = _expectation_lookup(profile)
    return [expectation_map[eid] for eid in stage.expectations if eid in expectation_map]


def _relevance_tokens_for_expectation(expectation: object) -> List[str]:
    raw = " ".join([str(getattr(expectation, "claim", ""))] + list(getattr(expectation, "evidence_queries", [])))
    tokens = [token for token in re.findall(r"[a-z0-9]+", raw.lower()) if len(token) >= 4]
    filtered = [token for token in tokens if token not in RELEVANCE_STOPWORDS]
    return list(dict.fromkeys(filtered))


def _relevant_citations_for_expectation(expectation: object, citations: List[Citation], limit: int = 2) -> List[Citation]:
    if not citations:
        return []
    tokens = _relevance_tokens_for_expectation(expectation)
    if not tokens:
        return citations[:limit]

    ranked: List[tuple[int, float, Citation]] = []
    for citation in citations:
        text = citation.snippet.lower()
        hits = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", text))
        retrieval_score = float(citation.retrieval_score or 0.0)
        ranked.append((hits, retrieval_score, citation))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    with_hits = [citation for hits, _, citation in ranked if hits > 0]
    if with_hits:
        return with_hits[:limit]
    return [ranked[0][2]]


def evaluate_profile_expectations(
    *,
    snapshot: CompanyPeopleSnapshot,
    profile: ConsultantProfile,
    stage_id: str,
    stage_label: str,
    evidence_result: Dict[str, object],
) -> Tuple[List[Finding], List[RiskItem], List[str]]:
    findings: List[Finding] = []
    risks: List[RiskItem] = []
    unknowns: List[str] = []

    expectation_evidence = evidence_result.get("expectation_evidence", {})
    expectation_statuses = evidence_result.get("expectation_statuses", {})

    for expectation in _active_expectations(profile, stage_id):
        rows = expectation_evidence.get(expectation.id, []) if isinstance(expectation_evidence, dict) else []
        retrieval_status = (
            expectation_statuses.get(expectation.id) if isinstance(expectation_statuses, dict) else None
        )
        citations, absence_rows = _to_citations_with_absence(rows if isinstance(rows, list) else [])
        citations = _relevant_citations_for_expectation(expectation, citations, limit=2)

        verdict = "not_found"
        analysis_ran = True
        if absence_rows and retrieval_status == "MENTIONED_EXPLICIT":
            verdict = "explicitly_missing"
            citations = _relevant_citations_for_expectation(expectation, absence_rows, limit=2)
        elif citations and retrieval_status == "MENTIONED_EXPLICIT":
            # Explicitly present expectations are tracked in coverage; not added as concern findings.
            continue
        elif retrieval_status == "NOT_RETRIEVED":
            analysis_ran = False
            verdict = "not_found"
        elif retrieval_status in {"MENTIONED_IMPLICIT", "MENTIONED_AMBIGUOUS"}:
            verdict = "ambiguous"
        elif citations:
            verdict = "ambiguous"

        evidence_status = map_verdict_to_evidence_status(
            verdict=verdict,
            retrieval_status=retrieval_status,
            has_relevant_evidence=bool(citations),
            analysis_ran=analysis_ran,
        )

        if evidence_status == "explicitly_missing":
            stage_reason = f"At stage {stage_label}, reviewed evidence explicitly indicates this control is missing."
        elif evidence_status == "ambiguous":
            stage_reason = f"At stage {stage_label}, evidence was indirect/partial and needs confirmation."
        elif evidence_status == "not_assessed":
            stage_reason = f"At stage {stage_label}, expected source coverage for this control was not retrieved."
        else:
            stage_reason = f"At stage {stage_label}, this control was not found in reviewed sources."

        needs_confirmation = evidence_status_requires_confirmation(evidence_status)
        severity = expectation.severity_if_missing
        if needs_confirmation and retrieval_status == "NOT_RETRIEVED" and not expectation.compliance:
            severity = _severity_down_one(severity)

        followups = []
        if needs_confirmation and profile.ask_questions_when_unknown:
            question = expectation.follow_up_question or f"Can you confirm status of: {expectation.claim}?"
            followups.append(question)

        actions = [
            f"Define owner and baseline for: {expectation.claim}",
            f"Document and communicate operating standard for: {expectation.claim}",
        ]
        if needs_confirmation:
            actions = [f"Conditional: if this is not already in place, {action.lower()}" for action in actions]
        supporting_citations = citations if evidence_status in {"ambiguous", "explicitly_missing"} else []

        finding = Finding(
            check_id=f"profile:{expectation.id}",
            area=expectation.area,
            title=expectation.claim,
            severity=severity,
            evidence_status=evidence_status,
            retrieval_status=retrieval_status,
            needs_confirmation=needs_confirmation,
            is_threshold_prompt=bool(expectation.compliance),
            stage_reason=stage_reason,
            evidence=supporting_citations,
            subchecks=[
                Subcheck(
                    capability_key=expectation.id,
                    evidence_status=evidence_status,
                    retrieval_status=retrieval_status,
                    citations=(
                        supporting_citations
                        if evidence_status in {"present", "explicitly_missing", "ambiguous"}
                        else []
                    ),
                    missing_reason=(
                        "Expected sources were not retrieved."
                        if retrieval_status == "NOT_RETRIEVED"
                        else "Evidence is indirect/weak and requires explicit confirmation."
                        if evidence_status == "ambiguous"
                        else "Not explicitly confirmed in reviewed sources."
                    ),
                )
            ],
            actions=actions,
            owner=_owner_from_citations(supporting_citations),
            metrics=[],
            questions=followups,
        )
        findings.append(finding)

        if evidence_status in {"ambiguous", "not_provided_in_sources", "not_assessed"}:
            unknowns.append(expectation.id)

        if expectation.compliance or severity in {"high", "critical"}:
            risks.append(
                RiskItem(
                    category=expectation.area,
                    severity=severity,
                    statement=(
                        f"{expectation.claim}: explicit evidence of missing control."
                        if evidence_status == "explicitly_missing"
                        else f"{expectation.claim}: evidence is indirect/ambiguous and requires confirmation."
                        if evidence_status == "ambiguous"
                        else f"{expectation.claim}: control not found in reviewed sources."
                    ),
                    evidence=supporting_citations,
                    mitigation=actions[:2],
                )
            )

    return findings, risks, list(dict.fromkeys(unknowns))
