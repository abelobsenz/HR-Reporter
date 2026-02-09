import json
from pathlib import Path

import jsonschema
import pytest

from app.models import FinalReport, Plan3090

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_snapshot_golden_files_validate_schema() -> None:
    schema = _load_json(REPO_ROOT / "app" / "schemas" / "snapshot_schema.json")
    for fixture in sorted((REPO_ROOT / "tests" / "golden").glob("expected_snapshot_*.json")):
        payload = _load_json(fixture)
        jsonschema.validate(instance=payload, schema=schema)


def test_report_model_sample_validates_schema() -> None:
    schema = _load_json(REPO_ROOT / "app" / "schemas" / "report_schema.json")
    payload = FinalReport(
        stage="11-40 employees",
        confidence=0.72,
        drivers=["Headcount signal identified from reviewed sources."],
        plan_30_60_90=Plan3090(why_now="Reduce scale risk through foundational controls."),
    ).model_dump(mode="json")
    jsonschema.validate(instance=payload, schema=schema)


def test_default_pack_validates_pack_schema() -> None:
    yaml = pytest.importorskip("yaml")
    schema = _load_json(REPO_ROOT / "app" / "schemas" / "pack_schema.json")
    with (REPO_ROOT / "tuning" / "packs" / "default_pack.yaml").open("r", encoding="utf-8") as f:
        pack_payload = yaml.safe_load(f)
    jsonschema.validate(instance=pack_payload, schema=schema)
