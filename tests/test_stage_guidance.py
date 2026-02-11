from __future__ import annotations

from pathlib import Path

from app.logic.stage_guidance import build_stage_based_recommendation
from app.logic.stage_inference import infer_stage
from app.models import CompanyPeopleSnapshot, TextChunk
from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage_inference_detects_size_and_funding_stage_from_text() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    snapshot = CompanyPeopleSnapshot()
    chunks = [
        TextChunk(
            chunk_id="c1",
            doc_id="d1",
            section="notes",
            text="The company recently closed a Series B financing and has about 82 employees.",
        )
    ]

    result = infer_stage(snapshot=snapshot, profile=profile, chunks=chunks)

    assert result.stage_id == "50-99"
    assert result.funding_stage_id == "series_b"
    assert result.company_stage_label == "Series B | 50-99 employees"


def test_stage_guidance_returns_practices_and_sources() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")

    recommendation = build_stage_based_recommendation(
        profile=profile,
        size_stage_id="20-49",
        size_stage_label="20-49 employees",
        funding_stage_id="series_a",
        funding_stage_label="Series A",
    )

    assert recommendation.hr_structure_recommendation
    assert len(recommendation.recommended_practices) >= 2
    assert len(recommendation.potential_risks) >= 1
    assert len(recommendation.sources) >= 1
