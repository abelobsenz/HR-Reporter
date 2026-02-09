from __future__ import annotations

from typing import Any, Dict

from app.llm.extract_snapshot import extract_snapshot
from app.models import TextChunk


class _NullClient:
    def structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        schema_name: str,
    ) -> Dict[str, Any]:
        return {
            "evidence_map": [
                {
                    "field_path": "hris_data.hris_system",
                    "status": "not_provided_in_sources",
                    "citations": [],
                }
            ]
        }


def test_targeted_retrieval_overlay_recovers_anchored_evidence() -> None:
    chunk = TextChunk(
        chunk_id="doc-001-c001",
        doc_id="doc-001",
        section="General",
        start_char=10,
        end_char=190,
        text=(
            "Employees enter time away requests in Workday after manager approval. "
            "The HR operations team audits the Workday log monthly for completeness."
        ),
    )
    targeted = {
        "field_statuses": {"hris_data.hris_system": "MENTIONED_EXPLICIT"},
        "field_candidates": {
            "hris_data.hris_system": [
                {
                    "chunk_id": "doc-001-c001",
                    "doc_id": "doc-001",
                    "source_id": "url:https://example.com/hris",
                    "source": "https://example.com/hris",
                    "score": 8.2,
                    "snippet": "Employees enter time away requests in Workday after manager approval.",
                }
            ]
        },
        "coverage_notes": {"hris_data.hris_system": "Coverage matched expected pattern 'hris'."},
        "queries_by_field": {"hris_data.hris_system": ["workday", "hris", "time away"]},
    }

    snapshot = extract_snapshot(
        client=_NullClient(),
        chunks=[chunk],
        tracked_fields=["hris_data.hris_system"],
        targeted_retrieval=targeted,
    )
    assert snapshot.evidence_map["hris_data.hris_system"].status == "present"
    assert snapshot.evidence_map["hris_data.hris_system"].retrieval_status == "MENTIONED_EXPLICIT"
    citation = snapshot.evidence_map["hris_data.hris_system"].citations[0]
    assert citation.chunk_id == "doc-001-c001"
    assert citation.doc_id == "doc-001"
    assert citation.start_char == 10
    assert citation.end_char == 190


def test_targeted_retrieval_overlay_keeps_ambiguous_context_citation() -> None:
    chunk = TextChunk(
        chunk_id="doc-001-c001",
        doc_id="doc-001",
        section="General",
        start_char=25,
        end_char=210,
        text=(
            "Managers discuss time-off requests during weekly syncs and note policy references in shared docs. "
            "The page mentions overtime and leave considerations for managers across locations."
        ),
    )
    targeted = {
        "field_statuses": {"policies.overtime_policy": "MENTIONED_AMBIGUOUS"},
        "field_candidates": {
            "policies.overtime_policy": [
                {
                    "chunk_id": "doc-001-c001",
                    "doc_id": "doc-001",
                    "source_id": "url:https://example.com/policy",
                    "source": "https://example.com/policy",
                    "score": 3.4,
                    "snippet": "The page mentions overtime and leave considerations for managers across locations.",
                }
            ]
        },
        "coverage_notes": {"policies.overtime_policy": "Coverage matched expected pattern 'policy'."},
        "queries_by_field": {"policies.overtime_policy": ["overtime", "policy", "exempt"]},
    }

    snapshot = extract_snapshot(
        client=_NullClient(),
        chunks=[chunk],
        tracked_fields=["policies.overtime_policy"],
        targeted_retrieval=targeted,
    )
    status = snapshot.evidence_map["policies.overtime_policy"]
    assert status.status == "not_provided_in_sources"
    assert status.retrieval_status == "MENTIONED_AMBIGUOUS"
    assert len(status.citations) == 1
    citation = status.citations[0]
    assert citation.chunk_id == "doc-001-c001"
    assert citation.doc_id == "doc-001"
