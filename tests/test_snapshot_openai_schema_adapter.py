from __future__ import annotations

from app.llm.extract_snapshot import (
    _coerce_evidence_map_entries_to_dict,
    _snapshot_schema_for_openai,
)


def test_snapshot_schema_for_openai_uses_array_evidence_map() -> None:
    schema = _snapshot_schema_for_openai()
    evidence_map_schema = schema["properties"]["evidence_map"]
    assert evidence_map_schema["type"] == "array"
    assert "items" in evidence_map_schema


def test_coerce_evidence_map_entries_to_dict() -> None:
    payload = {
        "company_name": "X",
        "headcount": None,
        "headcount_range": None,
        "primary_locations": [],
        "current_priorities": [],
        "policies": {},
        "hiring": {},
        "onboarding": {},
        "manager_enablement": {},
        "performance": {},
        "comp_leveling": {},
        "hris_data": {},
        "er_retention": {},
        "benefits": {},
        "key_risks": [],
        "evidence_map": [
            {
                "field_path": "headcount",
                "status": "present",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "snippet": "The company currently has about 25 employees, and staffing is expected to grow this quarter.",
                    }
                ],
            }
        ],
    }
    coerced = _coerce_evidence_map_entries_to_dict(payload)
    assert isinstance(coerced["evidence_map"], dict)
    assert coerced["evidence_map"]["headcount"]["status"] == "present"


def test_menu_like_short_citation_is_not_allowed_for_present_status() -> None:
    payload = {
        "evidence_map": [
            {
                "field_path": "policies.leave_policy",
                "status": "present",
                "citations": [
                    {
                        "chunk_id": "c1",
                        "snippet": "Time Off and Absence",
                    }
                ],
            }
        ]
    }
    coerced = _coerce_evidence_map_entries_to_dict(payload)
    assert coerced["evidence_map"]["policies.leave_policy"]["status"] == "not_provided_in_sources"
    assert coerced["evidence_map"]["policies.leave_policy"]["citations"] == []
