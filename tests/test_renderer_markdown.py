from __future__ import annotations

from pathlib import Path

from app.config_loader import load_consultant_pack
from app.models import Citation, FinalReport, Finding, MetricItem, Plan3090, PlanAction, RiskItem, Subcheck
from app.report.renderer import render_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_report() -> FinalReport:
    finding = Finding(
        check_id="cp_employee_handbook",
        area="compliance_policies",
        title="Current Employee Handbook",
        severity="medium",
        evidence_status="not_provided_in_sources",
        needs_confirmation=True,
        is_threshold_prompt=False,
        stage_reason="At stage 11-40 employees, this capability was not provided in sources; confirm whether it exists.",
        evidence=[],
        subchecks=[
            Subcheck(
                capability_key="policies.employee_handbook",
                evidence_status="not_provided_in_sources",
                citations=[],
                missing_reason="Not provided in sources; confirm whether it exists.",
            )
        ],
        actions=["Conditional: if employee handbook is not in place, Draft handbook v1."],
        owner="TBD / assign (e.g., HR/People, Finance, Ops)",
        metrics=[
            MetricItem(metric="policies.employee_handbook", value=None, status="unknown", evidence=[]),
            MetricItem(metric="hris_data.mandatory_training_tracking", value="True", status="found", evidence=[]),
        ],
        questions=["Do you have a current employee handbook and last review date?"],
    )

    return FinalReport(
        stage="11-40 employees",
        confidence=0.74,
        drivers=["Headcount explicitly stated as 22."],
        top_growth_areas=[finding],
        risks=[
            RiskItem(
                category="compliance_policies",
                severity="medium",
                statement="Current Employee Handbook: Not provided in sources; confirm whether this control exists.",
                evidence=[],
                mitigation=["Conditional: if employee handbook is not in place, Draft handbook v1."],
            )
        ],
        plan_30_60_90=Plan3090(
            why_now="Foundation controls reduce near-term compliance risk.",
            days_30=[
                PlanAction(
                    action="Conditional: if prerequisite capability is not in place, Draft handbook baseline.",
                    rationale="Owner: TBD / assign",
                    evidence=[Citation(chunk_id="not_found", snippet="not found")],
                )
            ],
        ),
        follow_up_questions=["Do you have a current employee handbook and last review date?"],
        unknowns=["Not provided in sources: policies.employee_handbook"],
        reviewed_sources=["https://example.com/people"],
        stage_note="Stage inferred from proxies; explicit headcount not found.",
        coverage_note="Sources reviewed were primarily onboarding/time-off pages; other domains may not be covered.",
    )


def test_renderer_matches_golden_snapshot() -> None:
    pack = load_consultant_pack(REPO_ROOT / "tuning" / "packs" / "default_pack.yaml")
    rendered = render_markdown(_sample_report(), pack, repo_root=REPO_ROOT)
    expected = (REPO_ROOT / "tests" / "golden" / "rendered_report_snapshot.md").read_text(encoding="utf-8")

    normalize = lambda s: "\n".join(line.rstrip() for line in s.strip().splitlines())
    assert normalize(rendered) == normalize(expected)

    assert "Owner: Owner:" not in rendered
    assert "Evidence:\n- [" in rendered or "Evidence: Not provided in sources." in rendered
