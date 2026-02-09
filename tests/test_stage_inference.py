from pathlib import Path

from app.config_loader import load_consultant_pack
from app.logic.stage_inference import infer_stage
from app.models import CompanyPeopleSnapshot, TextChunk


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pack():
    return load_consultant_pack(REPO_ROOT / "tuning" / "packs" / "default_pack.yaml")


def test_stage_inference_from_headcount() -> None:
    snapshot = CompanyPeopleSnapshot(headcount=35)
    result = infer_stage(snapshot=snapshot, pack=_pack())
    assert result.stage_id == "11-40"
    assert result.stage_label == "11-40 employees"
    assert result.source == "rules"


def test_stage_inference_from_range() -> None:
    snapshot = CompanyPeopleSnapshot(headcount_range="120-180")
    result = infer_stage(snapshot=snapshot, pack=_pack())
    assert result.stage_id == "101-250"
    assert result.source == "rules"


def test_stage_inference_unknown_defaults_to_first_stage() -> None:
    snapshot = CompanyPeopleSnapshot()
    result = infer_stage(snapshot=snapshot, pack=_pack())
    assert result.stage_id == "0-10"
    assert result.source == "unknown"
    assert result.confidence <= 0.3


def test_stage_inference_global_proxy_signals_must_not_select_251_500() -> None:
    snapshot = CompanyPeopleSnapshot(
        current_priorities=[
            "global entities rollout",
            "board readiness",
            "investor relations cadence",
            "workday transformation",
            "multi-country payroll alignment",
        ]
    )
    result = infer_stage(snapshot=snapshot, pack=_pack())
    assert result.stage_id != "500+"
    assert result.stage_max == "500+"
    assert result.stage_point_estimate != "500+"
    assert result.note == "Stage inferred from proxies; explicit headcount not found."


def test_stage_inference_from_chunk_headcount_without_snapshot_field() -> None:
    snapshot = CompanyPeopleSnapshot()
    chunks = [
        TextChunk(
            chunk_id="doc-001-c001",
            doc_id="doc-001",
            section="Company overview",
            text="Current headcount is 62 employees across CA and NY.",
        )
    ]
    result = infer_stage(snapshot=snapshot, pack=_pack(), chunks=chunks)
    assert result.stage_id == "41-100"
    assert result.source == "rules"
    assert result.explicit_headcount_evidence is True
    assert len(result.stage_evidence) == 1


def test_stage_inference_ignores_30_60_90_style_numbers() -> None:
    snapshot = CompanyPeopleSnapshot()
    chunks = [
        TextChunk(
            chunk_id="doc-001-c001",
            doc_id="doc-001",
            section="Onboarding",
            text="New hires follow a 30/60/90 plan and should provide 30 days notice for long leave.",
        )
    ]
    result = infer_stage(snapshot=snapshot, pack=_pack(), chunks=chunks)
    assert result.stage_id == "0-10"
    assert result.source == "unknown"
