from __future__ import annotations

from typing import Literal

EvidenceSemantic = Literal[
    "present",
    "ambiguous",
    "not_provided_in_sources",
    "explicitly_missing",
    "not_assessed",
]

EVIDENCE_STATUS_PRECEDENCE: dict[str, int] = {
    "explicitly_missing": 5,
    "present": 4,
    "ambiguous": 3,
    "not_provided_in_sources": 2,
    "not_assessed": 1,
}


def map_verdict_to_evidence_status(
    *,
    verdict: str,
    retrieval_status: str | None = None,
    has_relevant_evidence: bool,
    analysis_ran: bool = True,
) -> EvidenceSemantic:
    normalized = (verdict or "").strip().lower()
    retrieval = (retrieval_status or "").strip().upper()

    if not analysis_ran:
        return "not_assessed"

    if normalized == "present":
        return "present"
    if normalized == "explicitly_missing":
        return "explicitly_missing"
    if normalized in {"ambiguous", "conflict"}:
        return "ambiguous" if has_relevant_evidence else "not_provided_in_sources"
    if normalized == "not_found":
        if retrieval == "NOT_RETRIEVED":
            return "not_assessed"
        return "not_provided_in_sources"
    return "not_provided_in_sources"


def evidence_status_requires_confirmation(status: str) -> bool:
    return status in {"ambiguous", "not_provided_in_sources", "not_assessed"}
