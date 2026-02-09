from __future__ import annotations

from pathlib import Path
from typing import Any

from app.llm.client import _normalize_schema_for_openai_strict
from app.utils import load_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_object_required_matches_properties(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            properties = node["properties"]
            required = node.get("required")
            assert isinstance(required, list)
            assert set(required) == set(properties.keys())
        for value in node.values():
            _assert_object_required_matches_properties(value)
        return

    if isinstance(node, list):
        for item in node:
            _assert_object_required_matches_properties(item)


def test_snapshot_schema_normalizes_for_openai_strict() -> None:
    schema = load_json(REPO_ROOT / "app" / "schemas" / "snapshot_schema.json")
    normalized = _normalize_schema_for_openai_strict(schema)
    _assert_object_required_matches_properties(normalized)
