from __future__ import annotations

from app.ingest.text_normalize import normalize_text


def test_normalize_removes_control_chars_and_keeps_layout() -> None:
    raw = "Policy\x00 text\x1f with\tcontrols\r\nand spacing.\r\rNext line."
    normalized = normalize_text(raw)
    assert "\x00" not in normalized
    assert "\x1f" not in normalized
    assert "\r" not in normalized
    assert "Policy text with controls" in normalized
    assert "\n\n" in normalized


def test_normalize_fixes_common_mojibake() -> None:
    raw = "GitLabâ€™s policy states benefits are updated annually."
    normalized = normalize_text(raw)
    assert "GitLab's policy" in normalized
