from __future__ import annotations

from pathlib import Path

from app.logic.evidence_collector import EvidenceCollector
from app.models import RawDocument, TextChunk
from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _doc(doc_id: str, text: str) -> RawDocument:
    return RawDocument(
        doc_id=doc_id,
        source_id=f"url:https://example.com/{doc_id}",
        source=f"https://example.com/{doc_id}",
        source_type="url",
        text=text,
    )


def test_legal_privacy_only_chunk_is_filtered_out_from_hr_evidence() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")

    privacy_doc = _doc(
        "doc-privacy",
        "This privacy policy explains personal information processing, data controller obligations, and DPF rights.",
    )
    hr_doc = _doc(
        "doc-hr",
        "We have an anti-harassment policy, clear complaint channels, and mandatory training completion tracked in HRIS.",
    )

    chunks = [
        TextChunk(
            chunk_id="doc-privacy-c001",
            doc_id="doc-privacy",
            section="Privacy",
            text=privacy_doc.text,
            nav_score=0.0,
        ),
        TextChunk(
            chunk_id="doc-hr-c001",
            doc_id="doc-hr",
            section="People Policies",
            text=hr_doc.text,
            nav_score=0.0,
        ),
    ]

    result = EvidenceCollector(profile=profile, chunks=chunks, documents=[privacy_doc, hr_doc], client=None).collect()

    anti_harassment_rows = result["expectation_evidence"].get("anti_harassment_policy", [])
    anti_harassment_chunk_ids = {row.get("chunk_id") for row in anti_harassment_rows}

    assert "doc-hr-c001" in anti_harassment_chunk_ids
    assert "doc-privacy-c001" not in anti_harassment_chunk_ids

    all_used_chunk_ids = {
        row.get("chunk_id")
        for rows in result["field_evidence"].values()
        for row in rows
        if isinstance(row, dict)
    }
    assert "doc-privacy-c001" not in all_used_chunk_ids


def test_foundational_hr_chunk_still_routes_when_router_agent_disabled() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    hr_doc = _doc(
        "doc-manager",
        "Manager training is mandatory and every manager runs a one-on-one cadence with direct reports.",
    )
    chunks = [
        TextChunk(
            chunk_id="doc-manager-c001",
            doc_id="doc-manager",
            section="Manager Enablement",
            text=hr_doc.text,
            nav_score=0.0,
        )
    ]

    result = EvidenceCollector(profile=profile, chunks=chunks, documents=[hr_doc], client=None).collect()

    assert result["expectation_statuses"].get("manager_training") in {
        "MENTIONED_EXPLICIT",
        "MENTIONED_IMPLICIT",
    }


def test_data_retention_legal_copy_not_misrouted_to_hr_retention() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    legal_doc = _doc(
        "doc-legal-retention",
        "Data retention notice: we retain personal information for legal compliance and data protection obligations.",
    )
    chunks = [
        TextChunk(
            chunk_id="doc-legal-retention-c001",
            doc_id="doc-legal-retention",
            section="Data Retention Notice",
            text=legal_doc.text,
            nav_score=0.0,
        )
    ]

    result = EvidenceCollector(profile=profile, chunks=chunks, documents=[legal_doc], client=None).collect()

    retention_rows = result["expectation_evidence"].get("engagement_signals", [])
    assert retention_rows == []
    assert result["expectation_statuses"].get("engagement_signals") in {
        "NOT_FOUND_IN_RETRIEVED",
        "NOT_RETRIEVED",
    }


def test_hr_policy_with_mixed_privacy_language_still_surfaces() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    mixed_doc = _doc(
        "doc-mixed",
        "Our employee handbook includes an anti-harassment policy and complaint channel. We also describe personal information handling.",
    )
    chunks = [
        TextChunk(
            chunk_id="doc-mixed-c001",
            doc_id="doc-mixed",
            section="Handbook Summary",
            text=mixed_doc.text,
            nav_score=0.0,
        )
    ]

    result = EvidenceCollector(profile=profile, chunks=chunks, documents=[mixed_doc], client=None).collect()

    anti_rows = result["expectation_evidence"].get("anti_harassment_policy", [])
    anti_chunk_ids = {row.get("chunk_id") for row in anti_rows}
    assert "doc-mixed-c001" in anti_chunk_ids


def test_snippet_selection_prefers_complete_sentence_from_document_context() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    doc_text = (
        "Team members are encouraged to escalate concerns to more people within the company without repercussions. "
        "Manager training is mandatory for all supervisors."
    )
    manager_doc = _doc("doc-manager-context", doc_text)
    fragment_start = doc_text.find("to more people")
    chunks = [
        TextChunk(
            chunk_id="doc-manager-context-c001",
            doc_id="doc-manager-context",
            section="Values",
            text=doc_text[fragment_start:],
            start_char=fragment_start,
            end_char=len(doc_text),
            nav_score=0.0,
        )
    ]

    result = EvidenceCollector(profile=profile, chunks=chunks, documents=[manager_doc], client=None).collect()
    manager_rows = result["expectation_evidence"].get("manager_training", [])
    assert manager_rows
    top_snippet = str(manager_rows[0].get("snippet", "")).lower()
    assert "manager training is mandatory for all supervisors" in top_snippet


def test_anti_harassment_does_not_route_dashboard_reporting_noise() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    dashboard_doc = _doc(
        "doc-dashboard",
        (
            "Dashboards cover reporting for support response times and device management trends. "
            "Internal systems provide reporting and status views."
        ),
    )
    policy_doc = _doc(
        "doc-code-of-conduct",
        (
            "Our anti-harassment policy prohibits retaliation and includes complaint intake channels. "
            "Managers receive annual harassment training."
        ),
    )
    chunks = [
        TextChunk(
            chunk_id="doc-dashboard-c001",
            doc_id="doc-dashboard",
            section="Internal systems",
            text=dashboard_doc.text,
        ),
        TextChunk(
            chunk_id="doc-code-of-conduct-c001",
            doc_id="doc-code-of-conduct",
            section="Code of conduct",
            text=policy_doc.text,
        ),
    ]

    result = EvidenceCollector(
        profile=profile,
        chunks=chunks,
        documents=[dashboard_doc, policy_doc],
        client=None,
    ).collect()

    anti_rows = result["expectation_evidence"].get("anti_harassment_policy", [])
    anti_ids = {row.get("chunk_id") for row in anti_rows}
    assert "doc-code-of-conduct-c001" in anti_ids
    assert "doc-dashboard-c001" not in anti_ids


def test_headcount_fields_do_not_take_capacity_planning_noise() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    ladders_doc = _doc(
        "doc-job-ladders",
        (
            "Engineering ladders include planning expectations for manager of one behaviors. "
            "Capacity planning and staffing discussions are part of quarterly role calibration."
        ),
    )
    chunks = [
        TextChunk(
            chunk_id="doc-job-ladders-c001",
            doc_id="doc-job-ladders",
            section="Job ladders",
            text=ladders_doc.text,
        )
    ]

    result = EvidenceCollector(
        profile=profile,
        chunks=chunks,
        documents=[ladders_doc],
        client=None,
    ).collect()

    assert result["field_evidence"].get("headcount", []) == []
    assert result["field_evidence"].get("headcount_range", []) == []
    assert result["field_statuses"].get("headcount") in {"NOT_FOUND_IN_RETRIEVED", "NOT_RETRIEVED"}
    assert result["field_statuses"].get("headcount_range") in {"NOT_FOUND_IN_RETRIEVED", "NOT_RETRIEVED"}


def test_generic_reporting_token_only_match_is_excluded() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    generic_doc = _doc(
        "doc-generic-reporting",
        (
            "The system has reporting dashboards and clear reporting standards for operational metrics. "
            "This page documents analytics uptime targets and service quality metrics only."
        ),
    )
    chunks = [
        TextChunk(
            chunk_id="doc-generic-reporting-c001",
            doc_id="doc-generic-reporting",
            section="Systems",
            text=generic_doc.text,
        )
    ]

    result = EvidenceCollector(
        profile=profile,
        chunks=chunks,
        documents=[generic_doc],
        client=None,
    ).collect()

    anti_rows = result["expectation_evidence"].get("anti_harassment_policy", [])
    assert anti_rows == []
    assert result["expectation_statuses"].get("anti_harassment_policy") in {
        "NOT_FOUND_IN_RETRIEVED",
        "NOT_RETRIEVED",
    }
