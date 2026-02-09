from __future__ import annotations

from app.ingest.loaders import parse_url_candidates


def test_parse_url_candidates_splits_concatenated_urls() -> None:
    raw = "https://buffer.com/openhttps:/buffer.com/about"
    urls = parse_url_candidates(raw)
    assert "https://buffer.com/open" in urls
    assert "https://buffer.com/about" in urls


def test_parse_url_candidates_dedupes_and_normalizes() -> None:
    raw = "https:/example.com/a https://example.com/a, http:/example.org/b"
    urls = parse_url_candidates(raw)
    assert urls == ["https://example.com/a", "http://example.org/b"]
