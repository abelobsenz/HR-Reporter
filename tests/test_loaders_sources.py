from __future__ import annotations

from pathlib import Path

from app.ingest.loaders import load_documents


def test_file_source_id_uses_relative_identifier(tmp_path: Path) -> None:
    file_path = tmp_path / "policies" / "handbook.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("Employee handbook policy text.", encoding="utf-8")

    docs = load_documents(input_path=tmp_path)
    file_doc = next(doc for doc in docs if doc.source_type == "file")

    assert file_doc.source_id is not None
    assert file_doc.source_id.startswith("file:")
    assert str(tmp_path) not in file_doc.source_id
    assert "/Users/" not in file_doc.source_id
    assert not file_doc.source.startswith("/")


def test_pasted_text_source_label_is_professional() -> None:
    docs = load_documents(pasted_text="There are about 50-70 people working at this company.")
    text_doc = next(doc for doc in docs if doc.source_type == "text")

    assert text_doc.source == "Pasted input text"
    assert text_doc.source != "--text"


def test_markdown_file_title_prefers_h1(tmp_path: Path) -> None:
    file_path = tmp_path / "benefits-and-perks.md"
    file_path.write_text(
        "# Benefits and Perks\n\n## Time Off\n- Bereavement leave: 10 days paid leave.\n",
        encoding="utf-8",
    )

    docs = load_documents(input_path=tmp_path)
    file_doc = next(doc for doc in docs if doc.source_type == "file")

    assert file_doc.title == "Benefits and Perks"
