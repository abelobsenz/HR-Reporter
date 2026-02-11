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
