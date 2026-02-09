from pathlib import Path

import pytest

from app.config_loader import load_consultant_pack


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_pack_validates() -> None:
    pack = load_consultant_pack(REPO_ROOT / "tuning" / "packs" / "default_pack.yaml")
    assert len(pack.stages) == 6
    assert len(pack.checks) >= 12
    assert {check.area for check in pack.checks} >= {
        "compliance_policies",
        "hiring",
        "onboarding",
        "manager_enablement",
        "performance",
        "comp_leveling",
        "hris_data",
        "er_retention",
        "benefits",
    }


def test_pack_with_wrong_disclaimer_fails(tmp_path: Path) -> None:
    invalid_pack = tmp_path / "invalid_pack.yaml"
    invalid_pack.write_text(
        """
version: "1.0"
name: "Invalid"
disclaimer: "Different disclaimer"
stages:
  - id: "0-10"
    label: "0-10"
    min_headcount: 0
    max_headcount: 10
    description: "x"
weights:
  area_weights: {compliance_policies: 1.0}
  severity_weights: {low: 1.0, medium: 2.0, high: 3.0, critical: 4.0}
  compliance_multiplier: 1.2
  unknown_penalty: 0.1
templates:
  markdown_template: "app/report/templates/report.md.j2"
checks:
  - id: "c1"
    area: "compliance_policies"
    title: "t"
    description: "d"
    severity: "low"
    required_fields: []
    actions: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_consultant_pack(invalid_pack)
