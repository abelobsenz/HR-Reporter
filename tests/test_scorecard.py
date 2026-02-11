from __future__ import annotations

from app.logic.scorecard import build_functional_scorecard
from app.models import Finding


def _finding(area: str, severity: str, evidence_status: str) -> Finding:
    return Finding(
        check_id=f"{area}_{severity}",
        area=area,
        title="Test finding",
        severity=severity,
        evidence_status=evidence_status,
        stage_reason="Reason",
    )


def test_build_functional_scorecard_prioritizes_significant_high_risk_areas() -> None:
    findings = [
        _finding("compliance", "critical", "not_provided_in_sources"),
        _finding("systems", "high", "not_provided_in_sources"),
    ]

    rows = build_functional_scorecard(findings=findings)

    assert len(rows) >= 3
    top = rows[0]
    assert top.functional_area == "Legal and Compliance"
    assert top.impact_level == "Significant"


def test_build_functional_scorecard_normalizes_alias_areas() -> None:
    findings = [
        _finding("deib", "high", "not_provided_in_sources"),
        _finding("diversity_equity_inclusion_belonging", "medium", "not_provided_in_sources"),
    ]

    rows = build_functional_scorecard(findings=findings)
    labels = [row.functional_area for row in rows if row.rationale and "concern(s) were flagged" in row.rationale]
    assert labels.count("DEIB") == 1


def test_build_functional_scorecard_does_not_default_to_strategic_when_coverage_is_low() -> None:
    rows = build_functional_scorecard(
        findings=[],
        coverage_summary={
            "retrieved_chunks": 4,
            "fields_not_retrieved": 10,
            "fields_not_found": 8,
            "fields_with_explicit": 1,
        },
    )
    assert rows
    assert rows[0].maturity_level in {"Developing", "Aligned"}


def test_build_functional_scorecard_merges_performance_management_aliases() -> None:
    findings = [
        _finding("performance_mgmt", "high", "not_provided_in_sources"),
        _finding("Performance Management", "medium", "not_provided_in_sources"),
    ]

    rows = build_functional_scorecard(findings=findings)
    flagged_labels = [
        row.functional_area
        for row in rows
        if row.rationale and "concern(s) were flagged" in row.rationale
    ]
    assert flagged_labels.count("Performance Management") == 1
