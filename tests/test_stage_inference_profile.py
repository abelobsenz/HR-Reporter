from __future__ import annotations

from pathlib import Path

from app.logic.stage_inference import infer_stage
from app.models import CompanyPeopleSnapshot
from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage_inference_uses_profile_stages() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    snapshot = CompanyPeopleSnapshot(headcount=140)
    result = infer_stage(snapshot=snapshot, profile=profile)
    assert result.stage_id == "100-250"
    assert result.source == "rules"
    assert result.confidence == 1.0


def test_stage_inference_includes_missing_signals_and_alternate_candidates_when_uncertain() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    snapshot = CompanyPeopleSnapshot()
    result = infer_stage(snapshot=snapshot, profile=profile)
    assert result.confidence <= 0.68
    assert "explicit_headcount" in result.signals_missing
    assert len(result.candidates) >= 2
