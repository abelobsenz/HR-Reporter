from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict
import logging


logger = logging.getLogger(__name__)


class OpenAIResponsesClient:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        timeout_seconds: int = 90,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self._stats = {
            "responses_calls": 0,
            "responses_errors": 0,
        }

        if not self.api_key:
            self._client = None
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LLM calls") from exc

        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def is_enabled(self) -> bool:
        return self._client is not None

    def structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        schema_name: str,
    ) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide it to run LLM extraction and planning."
            )

        schema_for_openai = _normalize_schema_for_openai_strict(json_schema)

        try:
            logger.info("openai_request_start schema=%s model=%s", schema_name, self.model)
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema_for_openai,
                        "strict": True,
                    }
                },
            )
            self._stats["responses_calls"] = int(self._stats["responses_calls"]) + 1
            logger.info(
                "openai_request_ok schema=%s calls=%s",
                schema_name,
                self._stats["responses_calls"],
            )
        except Exception:
            self._stats["responses_errors"] = int(self._stats["responses_errors"]) + 1
            logger.exception("openai_request_error schema=%s", schema_name)
            raise

        output_text = self._extract_text(response)
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model returned non-JSON output: {output_text}") from exc

    def stats(self) -> Dict[str, int]:
        return {
            "responses_calls": int(self._stats.get("responses_calls", 0)),
            "responses_errors": int(self._stats.get("responses_errors", 0)),
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        # New SDK convenience field.
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        # Fallback path for raw output structure.
        output = getattr(response, "output", None)
        if not output:
            raise RuntimeError("No output returned from Responses API")

        for item in output:
            content = getattr(item, "content", None) or item.get("content", [])
            for block in content:
                block_type = getattr(block, "type", None) or block.get("type")
                if block_type in {"output_text", "text"}:
                    text = getattr(block, "text", None) or block.get("text")
                    if text:
                        return text

        raise RuntimeError("Could not extract text payload from Responses API response")


def _normalize_schema_for_openai_strict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust JSON Schema to satisfy OpenAI strict schema requirements.

    OpenAI strict mode requires object schemas with `properties` to include a
    `required` array that lists every property key.
    """
    normalized = deepcopy(schema)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                props: Dict[str, Any] = node["properties"]
                node["required"] = list(props.keys())
                # Strict object behavior is expected by this app's contracts.
                node.setdefault("additionalProperties", False)

            for value in node.values():
                _walk(value)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(normalized)
    return normalized
