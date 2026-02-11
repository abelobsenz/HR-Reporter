from __future__ import annotations

from typing import List
import logging

from app.models import Finding

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

EVIDENCE_STATUS_ORDER = {
    "explicitly_missing": 5,
    "ambiguous": 4,
    "not_assessed": 3,
    "not_provided_in_sources": 2,
    "present": 1,
}

RETRIEVAL_ORDER = {
    "MENTIONED_EXPLICIT": 5,
    "MENTIONED_IMPLICIT": 4,
    "MENTIONED_AMBIGUOUS": 3,
    "NOT_FOUND_IN_RETRIEVED": 2,
    "NOT_RETRIEVED": 1,
    None: 0,
}


def _driver_line_from_finding(finding: Finding) -> str:
    title = finding.title.strip().rstrip(".")
    area = finding.area.replace("_", " ")
    if finding.evidence_status == "explicitly_missing":
        return f"Sources indicate a required control is not in place in {area}."
    if finding.evidence_status == "ambiguous":
        return f"Evidence is partial/indirect for a high-priority control in {area}; confirm before finalizing."
    if finding.evidence_status == "not_provided_in_sources":
        return f"High-priority control evidence was not found in reviewed sources for {area}."
    if finding.evidence_status == "not_assessed":
        return f"Source retrieval coverage was insufficient to assess a high-priority control in {area}."
    return f"Evidence indicates this control is documented: {title}."


def build_growth_drivers(initial_drivers: List[str], findings: List[Finding], top_n: int = 3) -> List[str]:
    drivers = list(initial_drivers)
    contributed = 0
    for finding in findings:
        if contributed >= top_n:
            break
        if finding.evidence_status == "present" and finding.severity in {"low", "medium"}:
            continue
        drivers.append(_driver_line_from_finding(finding))
        contributed += 1

    deduped = []
    seen = set()
    for driver in drivers:
        if driver in seen:
            continue
        seen.add(driver)
        deduped.append(driver)
    logger.debug("drivers_built count=%s", len(deduped))
    return deduped


def rank_findings(findings: List[Finding]) -> List[Finding]:
    """Rank findings without numeric scoring.

    Ordering is deterministic:
    1) evidence status criticality (explicitly missing first),
    2) severity,
    3) retrieval explicitness,
    4) citation count.
    """

    ranked = sorted(
        findings,
        key=lambda item: (
            EVIDENCE_STATUS_ORDER.get(item.evidence_status, 0),
            SEVERITY_ORDER.get(item.severity, 0),
            RETRIEVAL_ORDER.get(item.retrieval_status, 0),
            len(item.evidence),
            item.title.lower(),
        ),
        reverse=True,
    )
    logger.info("finding_ranking_done findings=%s", len(ranked))
    return ranked
