from __future__ import annotations

from pathlib import Path

from app.config_loader import load_consultant_pack
from app.logic.checks_engine import run_checks
from app.models import CompanyPeopleSnapshot, EvidenceStatus, Citation, ManagerEnablement


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_explicit_false_with_citation_is_treated_as_known_gap() -> None:
    pack = load_consultant_pack(REPO_ROOT / "tuning" / "packs" / "default_pack.yaml")
    snapshot = CompanyPeopleSnapshot(
        manager_enablement=ManagerEnablement(manager_training=False),
        evidence_map={
            "manager_enablement.manager_training": EvidenceStatus(
                status="present",
                citations=[
                    Citation(
                        chunk_id="doc-001-c001",
                        snippet="No formal manager training yet.",
                    )
                ],
            )
        },
    )

    findings, _, unknowns = run_checks(
        snapshot=snapshot,
        pack=pack,
        stage_id="11-40",
        stage_label="11-40 employees",
    )

    mgr_finding = next(f for f in findings if f.check_id == "mgr_training")
    assert mgr_finding.evidence[0].chunk_id == "doc-001-c001"
    assert mgr_finding.evidence[0].snippet == "No formal manager training yet."
    assert mgr_finding.metrics[0].status == "found"
    assert mgr_finding.metrics[0].value == "False"
    assert "manager_enablement.manager_training" not in unknowns
