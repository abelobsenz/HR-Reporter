import json
from pathlib import Path

import jsonschema
from app.models import FinalReport

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
        drivers=["Headcount signal identified from reviewed sources."],
    ).model_dump(mode="json")
    jsonschema.validate(instance=payload, schema=schema)


def test_default_profile_validates_profile_schema() -> None:
    yaml = __import__("yaml")
    schema = _load_json(REPO_ROOT / "app" / "schemas" / "profile_schema.json")
    with (REPO_ROOT / "tuning" / "profile.yaml").open("r", encoding="utf-8") as f:
        profile_payload = yaml.safe_load(f)
    jsonschema.validate(instance=profile_payload, schema=schema)
