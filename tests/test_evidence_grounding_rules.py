from __future__ import annotations

from pathlib import Path

from app.config_loader import load_consultant_pack
from app.logic.checks_engine import run_checks
from app.models import Citation, CompanyPeopleSnapshot, EvidenceStatus, Finding, ManagerEnablement
from app.pipeline import _rank_follow_up_questions


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pack():
    return load_consultant_pack(REPO_ROOT / "tuning" / "packs" / "default_pack.yaml")


def test_not_provided_status_does_not_use_missing_wording() -> None:
    snapshot = CompanyPeopleSnapshot(headcount=20)
    findings, _, _ = run_checks(
        snapshot=snapshot,
        pack=_pack(),
        stage_id="11-40",
        stage_label="11-40 employees",
    )

    handbook = next(f for f in findings if f.check_id == "cp_employee_handbook")
    assert handbook.evidence_status == "not_provided_in_sources"
    assert handbook.needs_confirmation is True
    assert "not provided in sources" in handbook.stage_reason.lower()
    assert "missing" not in handbook.stage_reason.lower()
    assert handbook.severity == "medium"  # downgraded from high


def test_orphan_findings_removed_invariants_hold() -> None:
    snapshot = CompanyPeopleSnapshot(headcount=20)
    findings, _, _ = run_checks(
        snapshot=snapshot,
        pack=_pack(),
        stage_id="11-40",
        stage_label="11-40 employees",
    )

    assert findings
    for finding in findings:
        grounded = finding.evidence_status in {"present", "explicitly_missing"} and len(finding.evidence) > 0
        confirmable = finding.needs_confirmation and len(finding.questions) > 0
        assert grounded or confirmable


def test_follow_up_questions_capped_and_deduped() -> None:
    findings = [
        Finding(
            check_id=f"c{i}",
            area="hiring",
            title=f"F{i}",
            severity="high",
            evidence_status="not_provided_in_sources",
            needs_confirmation=True,
            stage_reason="Not provided in sources; confirm whether it exists.",
            questions=["Can you confirm whether an employee handbook exists?"],
        )
        for i in range(10)
    ]
    findings.extend(
        [
            Finding(
                check_id=f"x{i}",
                area="performance",
                title=f"X{i}",
                severity="medium",
                evidence_status="not_provided_in_sources",
                needs_confirmation=True,
                stage_reason="Not provided in sources; confirm whether it exists.",
                questions=[f"Is review cadence defined for team {i}?"],
            )
            for i in range(12)
        ]
    )

    ranked = _rank_follow_up_questions(findings, cap=8)
    assert len(ranked) <= 8
    assert len(ranked) == len(set(ranked))


def test_owner_not_hallucinated_when_not_supported() -> None:
    snapshot = CompanyPeopleSnapshot(
        manager_enablement=ManagerEnablement(manager_training=False),
        evidence_map={
            "manager_enablement.manager_training": EvidenceStatus(
                status="present",
                citations=[Citation(chunk_id="doc-1", snippet="No formal manager training yet.")],
            )
        },
    )

    findings, _, _ = run_checks(
        snapshot=snapshot,
        pack=_pack(),
        stage_id="11-40",
        stage_label="11-40 employees",
    )
    mgr = next(f for f in findings if f.check_id == "mgr_training")
    assert mgr.owner == "TBD / assign (e.g., HR/People, Finance, Ops)"


def test_ambiguous_retrieval_keeps_not_provided_with_context_citation() -> None:
    snapshot = CompanyPeopleSnapshot(
        evidence_map={
            "policies.overtime_policy": EvidenceStatus(
                status="not_provided_in_sources",
                retrieval_status="MENTIONED_AMBIGUOUS",
                citations=[
                    Citation(
                        chunk_id="doc-001-c010",
                        snippet="The policy page mentions overtime and leave context but does not define an eligibility rule.",
                    )
                ],
            )
        }
    )

    findings, _, unknowns = run_checks(
        snapshot=snapshot,
        pack=_pack(),
        stage_id="11-40",
        stage_label="11-40 employees",
    )
    overtime = next(f for f in findings if f.check_id == "cp_leave_overtime:policies.overtime_policy")
    assert overtime.evidence_status == "not_provided_in_sources"
    assert overtime.retrieval_status == "MENTIONED_AMBIGUOUS"
    assert overtime.evidence
    assert overtime.evidence[0].chunk_id == "doc-001-c010"
    assert "policies.overtime_policy" in unknowns
