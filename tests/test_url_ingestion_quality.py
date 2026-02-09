from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

from app.ingest.chunker import chunk_documents
from app.ingest.loaders import load_documents


@dataclass
class _FakeResponse:
    url: str
    text: str
    status_code: int = 200
    content_type: str = "text/html; charset=utf-8"

    @property
    def headers(self) -> Dict[str, str]:
        return {"content-type": self.content_type}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code} error")


def _long_sentence_block(topic: str) -> str:
    return (
        f"{topic} policy is documented in the handbook and applies across multiple teams. "
        f"This page explains implementation details, escalation guidance, and review cadence. "
        f"Managers are expected to follow the process and record outcomes in a consistent way. "
        f"The policy includes regional exceptions and points to accountable owners for updates."
    )


def test_directory_page_crawl_fetches_child_pages_and_yields_sentence_chunks(monkeypatch) -> None:
    root_url = "https://handbook.gitlab.com/handbook/people-policies/"
    child_time_off = "https://handbook.gitlab.com/handbook/people-group/time-off-and-absence/"
    child_harassment = "https://handbook.gitlab.com/handbook/people-group/anti-harassment/"

    html_by_url = {
        root_url: f"""
            <html><body>
              <nav>
                <a href="/handbook/people-group/time-off-and-absence/">Time Off</a>
                <a href="/handbook/people-group/anti-harassment/">Anti-Harassment</a>
                <a href="/handbook/hiring/">Hiring</a>
              </nav>
              <main><h1>People Policies</h1><p>Policy index</p></main>
            </body></html>
        """,
        child_time_off: f"""
            <html><body>
              <main><article>
                <h1>Time Off and Absence</h1>
                <p>{_long_sentence_block("Leave")}</p>
                <p>{_long_sentence_block("Absence management")}</p>
              </article></main>
            </body></html>
        """,
        child_harassment: f"""
            <html><body>
              <main><article>
                <h1>Anti-Harassment Policy</h1>
                <p>{_long_sentence_block("Anti-harassment")}</p>
                <p>{_long_sentence_block("Investigation")}</p>
              </article></main>
            </body></html>
        """,
        "https://handbook.gitlab.com/handbook/hiring/": f"""
            <html><body><main><article><p>{_long_sentence_block("Hiring")}</p></article></main></body></html>
        """,
    }

    def _fake_get(url: str, timeout: int = 15, allow_redirects: bool = True):  # noqa: ARG001
        if url not in html_by_url:
            raise RuntimeError(f"URL not mocked: {url}")
        return _FakeResponse(url=url, text=html_by_url[url])

    monkeypatch.setattr("requests.get", _fake_get)

    docs = load_documents(urls=[root_url])
    sources = {doc.source for doc in docs}
    assert child_time_off in sources
    assert child_harassment in sources

    chunks, _ = chunk_documents(docs)
    assert chunks, "Expected chunks from crawled child pages"

    for chunk in chunks:
        assert len(chunk.text) >= 300
        assert len(re.findall(r"[.!?](?:\s|$)", chunk.text)) >= 2

    joined = "\n".join(chunk.text.lower() for chunk in chunks)
    assert "leave policy is documented in the handbook" in joined
    assert "anti-harassment policy" in joined


def test_directory_crawl_is_disabled_when_seed_url_count_is_high(monkeypatch) -> None:
    root_a = "https://example.com/a"
    root_b = "https://example.com/b"
    child_a = "https://example.com/a/child-policy"
    html_by_url = {
        root_a: """
            <html><body>
              <nav><a href="/a/child-policy">Policy Child</a></nav>
              <main><p>Directory page for A.</p></main>
            </body></html>
        """,
        root_b: """
            <html><body><main><p>Second seed page with short content.</p></main></body></html>
        """,
        child_a: """
            <html><body><main><article><p>
            This child contains substantial policy detail, implementation steps, and owner guidance.
            It is intentionally long enough to be useful for extraction and chunking in tests.
            </p></article></main></body></html>
        """,
    }
    called_urls: list[str] = []

    def _fake_get(url: str, timeout: int = 15, allow_redirects: bool = True):  # noqa: ARG001
        called_urls.append(url)
        if url not in html_by_url:
            raise RuntimeError(f"URL not mocked: {url}")
        return _FakeResponse(url=url, text=html_by_url[url])

    monkeypatch.setattr("requests.get", _fake_get)
    monkeypatch.setenv("HR_REPORT_URL_CRAWL_MAX_SEED_URLS", "1")

    _ = load_documents(urls=[root_a, root_b])

    assert root_a in called_urls
    assert root_b in called_urls
    assert child_a not in called_urls


def test_directory_crawl_respects_total_fetch_budget(monkeypatch) -> None:
    root = "https://example.com/root"
    children = [f"https://example.com/root/policy-{idx}" for idx in range(1, 6)]
    root_links = "".join([f'<a href="/root/policy-{idx}">Policy {idx}</a>' for idx in range(1, 6)])
    html_by_url = {
        root: f"<html><body><nav>{root_links}</nav><main><p>Index page.</p></main></body></html>",
    }
    for child in children:
        html_by_url[child] = f"""
            <html><body><main><article><p>
            {child} includes detailed policy language, escalation paths, and governance responsibilities.
            This sentence exists so extraction returns meaningful sentence-rich content for tests.
            </p></article></main></body></html>
        """

    called_urls: list[str] = []

    def _fake_get(url: str, timeout: int = 15, allow_redirects: bool = True):  # noqa: ARG001
        called_urls.append(url)
        if url not in html_by_url:
            raise RuntimeError(f"URL not mocked: {url}")
        return _FakeResponse(url=url, text=html_by_url[url])

    monkeypatch.setattr("requests.get", _fake_get)
    monkeypatch.setenv("HR_REPORT_URL_CRAWL_MAX_SEED_URLS", "10")
    monkeypatch.setenv("HR_REPORT_URL_CRAWL_LIMIT", "25")
    monkeypatch.setenv("HR_REPORT_URL_MAX_TOTAL_FETCHES", "3")

    _ = load_documents(urls=[root])
    assert len(called_urls) <= 3
