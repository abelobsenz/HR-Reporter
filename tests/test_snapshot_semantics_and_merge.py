from __future__ import annotations

import json
from typing import Any, Dict, List

from app.llm.extract_snapshot import (
    _merge_snapshot_payloads,
    _select_relevant_evidence_rows,
    extract_snapshot_from_evidence,
)


class _AmbiguousAnalysisClient:
    def is_enabled(self) -> bool:
        return True

    def structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        schema_name: str,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        if schema_name == "snapshot_field_analysis":
            body = json.loads(user_prompt.split("\n", 1)[1])
            field = body["tracked_fields"][0]
            snippet = body["evidence_by_field"][field][0]["snippet"]
            chunk_id = body["evidence_by_field"][field][0]["chunk_id"]
            return {
                "field_assessments": [
                    {
                        "field_path": field,
                        "verdict": "ambiguous",
                        "support_strength": "weak",
                        "confidence": 0.41,
                        "value_text": "",
                        "hr_reasoning": "Evidence is indirect.",
                        "needs_confirmation": True,
                        "citations": [{"chunk_id": chunk_id, "snippet": snippet}],
                    }
                ]
            }
        return {
            "evidence_map": [
                {
                    "field_path": "policies.leave_policy",
                    "status": "present",
                    "citations": [],
                }
            ]
        }


def test_retrieval_status_does_not_override_analysis_semantic_status() -> None:
    snapshot = extract_snapshot_from_evidence(
        client=_AmbiguousAnalysisClient(),
        field_evidence={
            "policies.leave_policy": [
                {
                    "chunk_id": "doc-1-c001",
                    "snippet": "The handbook references time-off practices but no explicit leave eligibility matrix is defined.",
                    "retrieval_score": 8.0,
                    "kind": "implicit",
                }
            ]
        },
        tracked_fields=["policies.leave_policy"],
        retrieval_statuses={"policies.leave_policy": "MENTIONED_EXPLICIT"},
        retrieval_queries={"policies.leave_policy": ["leave policy"]},
    )
    status = snapshot.evidence_map["policies.leave_policy"]
    assert status.status == "ambiguous"
    assert status.retrieval_status == "MENTIONED_EXPLICIT"


def test_merge_snapshot_payloads_is_monotonic_for_values_and_statuses() -> None:
    payloads = [
        {
            "policies": {"leave_policy": True},
            "evidence_map": [
                {
                    "field_path": "policies.leave_policy",
                    "status": "present",
                    "citations": [
                        {
                            "chunk_id": "doc-1-c001",
                            "snippet": "The leave policy is documented and includes eligibility rules by location.",
                        }
                    ],
                }
            ],
        },
        {
            "policies": {"leave_policy": None},
            "evidence_map": [
                {
                    "field_path": "policies.leave_policy",
                    "status": "not_provided_in_sources",
                    "citations": [],
                }
            ],
        },
    ]
    merged = _merge_snapshot_payloads(payloads=payloads, tracked_fields=["policies.leave_policy"])
    assert merged["policies"]["leave_policy"] is True
    assert merged["evidence_map"]["policies.leave_policy"]["status"] == "present"


def test_relevance_filter_drops_role_ladder_noise_for_policy_fields() -> None:
    rows: List[Dict[str, Any]] = [
        {
            "chunk_id": "doc-1-c001",
            "snippet": "| Category | Junior (L1) | Senior (L3) | *Mastery* | *Scope* |",
            "retrieval_score": 9.9,
        },
        {
            "chunk_id": "doc-2-c004",
            "snippet": "The leave policy provides 20 PTO days and documents eligibility and approval rules by location.",
            "retrieval_score": 5.1,
        },
    ]
    selected = _select_relevant_evidence_rows(field_path="policies.leave_policy", rows=rows, limit=5)
    selected_ids = [row["chunk_id"] for row in selected]
    assert "doc-2-c004" in selected_ids
    assert "doc-1-c001" not in selected_ids
