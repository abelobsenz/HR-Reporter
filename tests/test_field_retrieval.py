from __future__ import annotations

from pathlib import Path

from app.config_loader import load_consultant_pack
from app.logic.field_retrieval import run_field_targeted_retrieval
from app.models import RawDocument, TextChunk


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pack():
    return load_consultant_pack(REPO_ROOT / "tuning" / "packs" / "default_pack.yaml")


def test_field_retrieval_finds_relevant_chunk() -> None:
    docs = [
        RawDocument(
            doc_id="doc-001",
            source_id="url:https://example.com/leave",
            source="https://example.com/leave-policy",
            source_type="url",
            text="Our leave policy explains sick time eligibility and required notice for all employees.",
        )
    ]
    chunks = [
        TextChunk(
            chunk_id="doc-001-c001",
            doc_id="doc-001",
            section="General",
            start_char=0,
            end_char=110,
            text=docs[0].text,
        )
    ]
    result = run_field_targeted_retrieval(
        chunks=chunks,
        documents=docs,
        tracked_fields=["policies.leave_policy"],
        pack=_pack(),
    )
    assert result["field_statuses"]["policies.leave_policy"] in {
        "MENTIONED_EXPLICIT",
        "MENTIONED_IMPLICIT",
    }
    assert result["field_candidates"]["policies.leave_policy"]


def test_field_retrieval_distinguishes_not_retrieved_vs_not_found() -> None:
    docs = [
        RawDocument(
            doc_id="doc-001",
            source_id="url:https://example.com/hiring",
            source="https://example.com/hiring",
            source_type="url",
            text="Team values and office setup guidance page.",
        )
    ]
    chunks = [
        TextChunk(
            chunk_id="doc-001-c001",
            doc_id="doc-001",
            section="General",
            start_char=0,
            end_char=30,
            text="Team values and office setup guidance page.",
        )
    ]

    result = run_field_targeted_retrieval(
        chunks=chunks,
        documents=docs,
        tracked_fields=["benefits.benefits_eligibility_policy", "hiring.structured_interviews"],
        pack=_pack(),
    )

    assert result["field_statuses"]["benefits.benefits_eligibility_policy"] == "NOT_RETRIEVED"
    assert result["field_statuses"]["hiring.structured_interviews"] == "NOT_FOUND_IN_RETRIEVED"
