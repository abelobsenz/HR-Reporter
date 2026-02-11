from __future__ import annotations

from pathlib import Path

from app.logic.profile_evaluator import evaluate_profile_expectations
from app.models import CompanyPeopleSnapshot
from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_does_not_phrase_does_not_force_explicitly_missing_subcheck() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    snapshot = CompanyPeopleSnapshot()
    evidence_result = {
        "expectation_evidence": {
            "anti_harassment_policy": [
                {
                    "chunk_id": "c-1",
                    "snippet": "The policy does not include examples for every case.",
                }
            ]
        },
        "expectation_statuses": {
            "anti_harassment_policy": "MENTIONED_IMPLICIT",
        },
    }

    findings, _, _ = evaluate_profile_expectations(
        snapshot=snapshot,
        profile=profile,
        stage_id="lt20",
        stage_label="<20 employees",
        evidence_result=evidence_result,
    )

    anti = next((row for row in findings if row.check_id == "profile:anti_harassment_policy"), None)
    assert anti is not None
    assert anti.evidence_status != "explicitly_missing"


def test_not_in_place_phrase_maps_to_explicitly_missing() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    snapshot = CompanyPeopleSnapshot()
    evidence_result = {
        "expectation_evidence": {
            "manager_training": [
                {
                    "chunk_id": "c-2",
                    "snippet": "Manager training is not in place yet for the current team.",
                }
            ]
        },
        "expectation_statuses": {
            "manager_training": "MENTIONED_EXPLICIT",
        },
    }

    findings, _, _ = evaluate_profile_expectations(
        snapshot=snapshot,
        profile=profile,
        stage_id="lt20",
        stage_label="<20 employees",
        evidence_result=evidence_result,
    )

    manager = next((row for row in findings if row.check_id == "profile:manager_training"), None)
    assert manager is not None
    assert manager.evidence_status == "explicitly_missing"


def test_without_repercussions_phrase_is_not_treated_as_missing_control() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    snapshot = CompanyPeopleSnapshot()
    evidence_result = {
        "expectation_evidence": {
            "manager_training": [
                {
                    "chunk_id": "c-3",
                    "snippet": "Team members can raise concerns to more people within the company without repercussions.",
                    "kind": "implicit",
                }
            ]
        },
        "expectation_statuses": {
            "manager_training": "MENTIONED_IMPLICIT",
        },
    }

    findings, _, _ = evaluate_profile_expectations(
        snapshot=snapshot,
        profile=profile,
        stage_id="lt20",
        stage_label="<20 employees",
        evidence_result=evidence_result,
    )

    manager = next((row for row in findings if row.check_id == "profile:manager_training"), None)
    assert manager is not None
    assert manager.evidence_status != "explicitly_missing"
