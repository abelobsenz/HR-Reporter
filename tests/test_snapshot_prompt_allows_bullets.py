from __future__ import annotations

from app.llm.prompts import (
    SNAPSHOT_SYSTEM_PROMPT,
    build_snapshot_user_prompt,
)
from app.models import TextChunk


def test_snapshot_system_prompt_allows_policy_bullets_and_table_rows() -> None:
    lower = SNAPSHOT_SYSTEM_PROMPT.lower()
    assert "bullet/list item" in lower
    assert "table row" in lower
    assert "do not cite headings alone" in lower


def test_snapshot_user_prompt_includes_doc_title_source_and_heading_path() -> None:
    chunk = TextChunk(
        chunk_id="doc-002-benefits-c004",
        doc_id="doc-002-benefits",
        doc_title="Benefits and Perks",
        source="data/input_requests/example/files/benefits.md",
        section="Time Off",
        heading_path=["Benefits & Perks", "Time Off"],
        text="- Bereavement leave: 10 days paid leave for immediate family.",
    )

    prompt = build_snapshot_user_prompt(
        chunks=[chunk],
        tracked_fields=["policies.leave_policy"],
    )

    assert "doc_title=Benefits and Perks" in prompt
    assert "heading_path=Benefits & Perks > Time Off" in prompt
    assert "source=data/input_requests/example/files/benefits.md" in prompt
