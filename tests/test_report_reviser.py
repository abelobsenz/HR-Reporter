from __future__ import annotations

from app.report.reviser import revise_markdown_report


class _FakeClient:
    def __init__(self, *, enabled: bool = True, response: str = "") -> None:
        self._enabled = enabled
        self._response = response
        self.calls: list[dict] = []

    def is_enabled(self) -> bool:
        return self._enabled

    def text_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
        schema_name: str = "text_completion",
    ) -> str:
        self.calls.append(
            {
                "schema_name": schema_name,
                "max_output_tokens": max_output_tokens,
                "model_override": model_override,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self._response


def _sample_markdown() -> str:
    return (
        "# HR Assessment Report\n\n"
        "## Executive Summary\n"
        "- This are an awkward sentence with citation (`doc-001-c001`).\n\n"
        "## Disclaimer\n"
        "This is not legal advice.\n"
    )


def test_reviser_applies_valid_markdown_updates() -> None:
    original = _sample_markdown()
    revised = (
        "# HR Assessment Report\n\n"
        "## Executive Summary\n"
        "- This is a polished sentence with citation (`doc-001-c001`).\n\n"
        "## Disclaimer\n"
        "This is not legal advice.\n"
    )
    client = _FakeClient(response=revised)

    output, meta = revise_markdown_report(client=client, markdown=original, profile_name="Default")

    assert output.strip() == revised.strip()
    assert meta["applied"] is True
    assert meta["reason"] == "success"
    assert client.calls and client.calls[0]["schema_name"] == "report_markdown_reviser"


def test_reviser_rejects_heading_changes() -> None:
    original = _sample_markdown()
    revised = (
        "# HR Assessment Report\n\n"
        "## Executive Summary Updated\n"
        "- This is rewritten (`doc-001-c001`).\n"
    )
    client = _FakeClient(response=revised)

    output, meta = revise_markdown_report(client=client, markdown=original)

    assert output == original
    assert meta["applied"] is False
    assert meta["reason"] == "validation_failed"
    assert meta["validation"]["heading_changed"] is True


def test_reviser_unwraps_markdown_code_fence() -> None:
    original = _sample_markdown()
    fenced = (
        "```markdown\n"
        "# HR Assessment Report\n\n"
        "## Executive Summary\n"
        "- This is improved (`doc-001-c001`).\n\n"
        "## Disclaimer\n"
        "This is not legal advice.\n"
        "```"
    )
    client = _FakeClient(response=fenced)

    output, meta = revise_markdown_report(client=client, markdown=original)

    assert output.startswith("# HR Assessment Report")
    assert "```" not in output
    assert meta["applied"] is True


def test_reviser_skips_when_client_disabled() -> None:
    original = _sample_markdown()
    client = _FakeClient(enabled=False, response="")

    output, meta = revise_markdown_report(client=client, markdown=original)

    assert output == original
    assert meta["applied"] is False
    assert meta["reason"] == "client_disabled"
