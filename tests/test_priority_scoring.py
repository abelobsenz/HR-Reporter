from __future__ import annotations

from app.logic.scoring import rank_findings
from app.models import Finding


def _finding(area: str, severity: str, retrieval_status: str | None = None) -> Finding:
    return Finding(
        check_id=f"{area}:{severity}",
        area=area,
        title=f"{area}-{severity}",
        severity=severity,
        evidence_status="not_provided_in_sources",
        retrieval_status=retrieval_status,
        needs_confirmation=True,
        stage_reason="Needs confirmation",
        evidence=[],
        subchecks=[],
        actions=["Action"],
        owner="TBD / assign (e.g., HR/People, Finance, Ops)",
        metrics=[],
        questions=["Q?"],
    )


def test_explicit_missing_is_ranked_ahead_of_unconfirmed() -> None:
    findings = [
        _finding("compliance", "high", "NOT_FOUND_IN_RETRIEVED"),
        _finding("benefits", "medium", "MENTIONED_EXPLICIT"),
    ]
    findings[0].evidence_status = "explicitly_missing"
    findings[1].evidence_status = "not_provided_in_sources"

    ranked = rank_findings(findings)

    assert ranked[0].area == "compliance"


def test_severity_breaks_ties_with_same_evidence_status() -> None:
    explicit = _finding("compliance", "high", "MENTIONED_EXPLICIT")
    medium = _finding("compliance", "medium", "MENTIONED_EXPLICIT")
    explicit.evidence_status = "not_provided_in_sources"
    medium.evidence_status = "not_provided_in_sources"

    ranked = rank_findings([medium, explicit])

    assert ranked[0].severity == "high"
