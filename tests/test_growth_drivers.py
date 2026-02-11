from __future__ import annotations

from app.logic.scoring import build_growth_drivers
from app.models import Finding


def _finding(*, status: str, severity: str = "critical") -> Finding:
    return Finding(
        check_id="profile:leave_overtime_coverage",
        area="compliance",
        title="Leave and overtime policies are documented with eligibility rules.",
        severity=severity,  # type: ignore[arg-type]
        evidence_status=status,  # type: ignore[arg-type]
        stage_reason="x",
    )


def test_growth_drivers_do_not_state_documented_when_control_is_unresolved() -> None:
    drivers = build_growth_drivers(
        initial_drivers=[],
        findings=[_finding(status="not_provided_in_sources")],
        top_n=1,
    )
    assert len(drivers) == 1
    assert "documented" not in drivers[0].lower()
    assert "not found" in drivers[0].lower()
