from __future__ import annotations

import json

from app.llm.client import OpenAIResponsesClient


class _Node:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_extract_text_prefers_output_text() -> None:
    response = _Node(output_text='{"ok":true}', output=[])
    text = OpenAIResponsesClient._extract_text(response)
    assert json.loads(text) == {"ok": True}


def test_extract_text_ignores_reasoning_items_and_reads_message_text() -> None:
    response = _Node(
        output=[
            _Node(type="reasoning"),
            _Node(
                type="message",
                content=[_Node(type="output_text", text='{"routes":[]}')],
            ),
        ]
    )
    text = OpenAIResponsesClient._extract_text(response)
    assert json.loads(text) == {"routes": []}


def test_extract_text_supports_dict_payloads() -> None:
    response = _Node(
        output=[
            {
                "type": "message",
                "content": [
                    {
                        "type": "text",
                        "text": '{"value": 3}',
                    }
                ],
            }
        ]
    )
    text = OpenAIResponsesClient._extract_text(response)
    assert json.loads(text) == {"value": 3}


def test_extract_text_reads_output_json_block() -> None:
    response = _Node(
        output=[
            _Node(
                type="message",
                content=[_Node(type="output_json", json={"x": 1, "y": "z"})],
            )
        ]
    )
    text = OpenAIResponsesClient._extract_text(response)
    assert json.loads(text) == {"x": 1, "y": "z"}
