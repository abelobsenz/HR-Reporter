from __future__ import annotations

from app.llm.extract_snapshot import extract_snapshot_from_evidence
from app.models import TextChunk


class _DisabledClient:
    def is_enabled(self) -> bool:
        return False


def test_inexact_headcount_range_does_not_set_scalar_headcount() -> None:
    pasted_chunk_id = "doc-001-pasted-text-c001"
    snippet = "There are about 50-70 people working at this company."

    snapshot = extract_snapshot_from_evidence(
        client=_DisabledClient(),
        field_evidence={
            "headcount": [],
            "headcount_range": [
                {
                    "chunk_id": pasted_chunk_id,
                    "doc_id": "doc-001-pasted-text",
                    "snippet": snippet,
                    "kind": "explicit",
                    "retrieval_score": 9.4,
                }
            ],
        },
        tracked_fields=["headcount", "headcount_range"],
        retrieval_statuses={
            "headcount": "NOT_FOUND_IN_RETRIEVED",
            "headcount_range": "MENTIONED_EXPLICIT",
        },
        retrieval_queries={
            "headcount": ["headcount", "employees"],
            "headcount_range": ["about X-Y people"],
        },
        source_chunks=[
            TextChunk(
                chunk_id=pasted_chunk_id,
                doc_id="doc-001-pasted-text",
                section="Input text excerpt",
                text=snippet,
            )
        ],
    )

    assert snapshot.headcount is None
    assert snapshot.headcount_range == "50-70"
    assert snapshot.evidence_map["headcount_range"].citations
    assert snapshot.evidence_map["headcount_range"].citations[0].chunk_id == pasted_chunk_id
