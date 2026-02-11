from __future__ import annotations

from app.models import Citation, FinalReport, Finding, MetricItem, RiskItem, Subcheck
from app.report.renderer import render_markdown


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
        drivers=["Headcount explicitly stated as 22."],
        profile_expectations=[finding],
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
        follow_up_questions=["Do you have a current employee handbook and last review date?"],
        unknowns=["Not provided in sources: policies.employee_handbook"],
        reviewed_sources=["https://example.com/people"],
        stage_note="Stage inferred from proxies; explicit headcount not found.",
        coverage_note="Sources reviewed were primarily onboarding/time-off pages; other domains may not be covered.",
    )


def test_renderer_matches_golden_snapshot() -> None:
    rendered = render_markdown(_sample_report(), profile_name="Default Profile")

    assert "# HR Assessment Report" in rendered
    assert "## HR Functional Scorecard" in rendered
    assert "## Stage-Based Recommendations" in rendered
    assert "## Functional Area Deep-Dives" in rendered
    assert "## Assumptions and Data Limitations" in rendered
    assert "## Disclaimer" in rendered
    assert "Owner: Owner:" not in rendered


def test_renderer_displays_ambiguous_status_explicitly() -> None:
    finding = Finding(
        check_id="cp_leave_policy",
        area="compliance_policies",
        title="Leave and overtime policy coverage",
        severity="high",
        evidence_status="ambiguous",
        retrieval_status="MENTIONED_AMBIGUOUS",
        needs_confirmation=True,
        stage_reason="Evidence is partial and requires confirmation.",
        evidence=[
            Citation(
                chunk_id="doc-1-c002",
                snippet="Benefits are described, but overtime eligibility rules are not explicitly specified in the policy text.",
            )
        ],
        subchecks=[
            Subcheck(
                capability_key="leave_overtime_coverage",
                evidence_status="ambiguous",
                retrieval_status="MENTIONED_AMBIGUOUS",
                citations=[
                    Citation(
                        chunk_id="doc-1-c002",
                        snippet="Benefits are described, but overtime eligibility rules are not explicitly specified in the policy text.",
                    )
                ],
            )
        ],
        actions=["Confirm overtime eligibility language and update policy text."],
        owner="People Team",
    )
    report = FinalReport(
        stage="Provisional: Seed | <20 employees",
        drivers=["Evidence is partial/indirect for this control and needs confirmation: Leave and overtime policy coverage."],
        profile_expectations=[finding],
        signals_missing=["Explicit headcount statement"],
        stage_confidence=0.54,
        stage_provisional=True,
    )

    rendered = render_markdown(report, profile_name="Default Profile")

    assert "Evidence status:** ambiguous" in rendered
    assert "Stage confidence:** 0.54 (Provisional)" in rendered
