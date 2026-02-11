from __future__ import annotations

from pathlib import Path

from app.logic.stage_inference import infer_stage
from app.models import Citation, CompanyPeopleSnapshot, EvidenceStatus, StageBand
from app.profile_loader import load_consultant_profile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _stages() -> list[StageBand]:
    return [
        StageBand(id="lt20", label="<20 employees", min_headcount=0, max_headcount=19, description=""),
        StageBand(id="20-49", label="20-49 employees", min_headcount=20, max_headcount=49, description=""),
        StageBand(id="50-99", label="50-99 employees", min_headcount=50, max_headcount=99, description=""),
        StageBand(id="100-250", label="100-250 employees", min_headcount=100, max_headcount=250, description=""),
    ]


def test_headcount_range_only_uses_range_driver_without_explicit_scalar_claim() -> None:
    snapshot = CompanyPeopleSnapshot(
        headcount=None,
        headcount_range="50-70",
        evidence_map={
            "headcount_range": EvidenceStatus(
                status="present",
                retrieval_status="MENTIONED_EXPLICIT",
                citations=[Citation(chunk_id="doc-1-c001", snippet="There are about 50-70 people in the company.")],
            )
        },
    )

    result = infer_stage(snapshot=snapshot, stages=_stages())

    assert result.stage_id == "50-99"
    assert result.stage_confidence == 0.72
    assert any("Headcount range signal found" in driver for driver in result.drivers)
    assert all("explicitly stated as" not in driver.lower() for driver in result.drivers)


def test_funding_fallback_prevents_perfect_stage_confidence() -> None:
    profile = load_consultant_profile(REPO_ROOT / "tuning" / "profile.yaml")
    snapshot = CompanyPeopleSnapshot(
        headcount=80,
        evidence_map={
            "headcount": EvidenceStatus(
                status="present",
                retrieval_status="MENTIONED_EXPLICIT",
                citations=[Citation(chunk_id="doc-1-c001", snippet="We have 80 employees in total.")],
            )
        },
    )

    result = infer_stage(snapshot=snapshot, profile=profile, chunks=[])

    assert result.stage_id == "50-99"
    assert result.funding_stage_confidence is not None
    assert result.funding_stage_evidence == []
    assert result.stage_confidence is not None
    assert result.stage_confidence < 1.0
