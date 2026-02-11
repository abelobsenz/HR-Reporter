from __future__ import annotations

from pathlib import Path

from app.ingest.loaders import load_documents


def test_load_documents_supports_csv_checklist(tmp_path: Path) -> None:
    csv_path = tmp_path / "checklist.csv"
    csv_path.write_text(
        "FUNCTIONAL AREA,Response,Comments\nPeople Strategy,Yes,In planning docs\n",
        encoding="utf-8",
    )

    docs = load_documents(input_path=tmp_path)
    assert len(docs) == 1
    text = docs[0].text
    assert "FUNCTIONAL AREA: People Strategy" in text
    assert "Response: Yes" in text
    assert "Comments: In planning docs" in text
