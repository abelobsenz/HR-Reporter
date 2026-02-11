from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
import time
import traceback
from typing import Any, Dict
import logging


logger = logging.getLogger(__name__)


class APICallTranscriptWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        run_id: str,
        model: str,
        profile_path: str,
        timeout_seconds: int | None,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._entry_counter = 0
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        timeout_label = "none" if timeout_seconds is None else str(timeout_seconds)
        header = (
            "# API Call Transcript\n\n"
            f"- run_id: `{run_id}`\n"
            f"- created_utc: `{datetime.now(timezone.utc).isoformat()}`\n"
            f"- profile_path: `{profile_path}`\n"
            f"- model: `{model}`\n"
            f"- timeout_seconds: `{timeout_label}`\n\n"
            "---\n"
        )
        self.path.write_text(header, encoding="utf-8")

    def log_event(self, *, title: str, details: str) -> None:
        self._append(
            f"\n## Event {self._next_entry_id()}: {title}\n\n"
            f"- timestamp_utc: `{datetime.now(timezone.utc).isoformat()}`\n"
            f"- details: {details}\n"
        )

    def log_call(
        self,
        *,
        schema_name: str,
        model: str,
        attempt: int,
        max_output_tokens: int | None,
        system_prompt: str,
        user_prompt: str,
        status: str,
        duration_ms: int,
        response_text: str | None = None,
        error_text: str | None = None,
    ) -> None:
        tokens = "none" if max_output_tokens is None else str(max_output_tokens)
        response_block = response_text if response_text is not None else ""
        error_block = error_text if error_text is not None else ""
        body = (
            f"\n## Call {self._next_entry_id()}: schema `{schema_name}` attempt `{attempt}`\n\n"
            f"- timestamp_utc: `{datetime.now(timezone.utc).isoformat()}`\n"
            f"- model: `{model}`\n"
            f"- max_output_tokens: `{tokens}`\n"
            f"- status: `{status}`\n"
            f"- duration_ms: `{duration_ms}`\n\n"
            "### System Prompt\n"
            "````text\n"
            f"{system_prompt}\n"
            "````\n\n"
            "### User Prompt\n"
            "````text\n"
            f"{user_prompt}\n"
            "````\n\n"
        )
        if response_text is not None:
            body += (
                "### Response\n"
                "````json\n"
                f"{response_block}\n"
                "````\n\n"
            )
        if error_text is not None:
            body += (
                "### Error\n"
                "````text\n"
                f"{error_block}\n"
                "````\n\n"
            )
        self._append(body)

    def _next_entry_id(self) -> int:
        with self._lock:
            self._entry_counter += 1
            return self._entry_counter

    def _append(self, content: str) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(content)


class OpenAIResponsesClient:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        timeout_seconds: int | None = 90,
        transcript_writer: APICallTranscriptWriter | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if timeout_seconds is not None and int(timeout_seconds) <= 0:
            timeout_seconds = None
        self.timeout_seconds = timeout_seconds
        self._transcript_writer = transcript_writer
        self._stats = {
            "responses_calls": 0,
            "responses_errors": 0,
        }
        self._disabled_reason: str | None = None

        if not self.api_key:
            self._client = None
            self._log_event("openai_client_disabled", "OPENAI_API_KEY not set.")
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LLM calls") from exc

        self._client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout_seconds,
            max_retries=0,
        )
        timeout_label = "none" if self.timeout_seconds is None else str(self.timeout_seconds)
        self._log_event("openai_client_enabled", f"timeout_seconds={timeout_label}")

    def is_enabled(self) -> bool:
        return self._client is not None

    def structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
    ) -> Dict[str, Any]:
        if not self._client:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide it to run LLM extraction and planning."
            )

        request_model = (model_override or "").strip() or self.model
        schema_for_openai = _normalize_schema_for_openai_strict(json_schema)
        call_attempt = 0

        def _request_once(tokens: int | None) -> Any:
            nonlocal call_attempt
            call_attempt += 1
            started = time.perf_counter()
            try:
                logger.info(
                    "openai_request_start schema=%s model=%s max_output_tokens=%s",
                    schema_name,
                    request_model,
                    tokens,
                )
                response_obj = self._client.responses.create(
                    model=request_model,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                    ],
                    max_output_tokens=tokens,
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
                duration_ms = int((time.perf_counter() - started) * 1000)
                self._log_call_attempt(
                    schema_name=schema_name,
                    attempt=call_attempt,
                    max_output_tokens=tokens,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    status="ok",
                    duration_ms=duration_ms,
                    model=request_model,
                    response_text=self._extract_text_for_transcript(response_obj),
                )
                return response_obj
            except Exception as exc:
                self._stats["responses_errors"] = int(self._stats["responses_errors"]) + 1
                logger.exception("openai_request_error schema=%s", schema_name)
                duration_ms = int((time.perf_counter() - started) * 1000)
                error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                self._log_call_attempt(
                    schema_name=schema_name,
                    attempt=call_attempt,
                    max_output_tokens=tokens,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    status="error",
                    duration_ms=duration_ms,
                    model=request_model,
                    error_text=error_text,
                )
                if self._looks_like_connection_error(exc):
                    self._disabled_reason = "connection_error"
                    self._client = None
                    logger.warning("openai_disabled reason=%s", self._disabled_reason)
                    self._log_event("openai_client_disabled", "reason=connection_error")
                raise

        response = _request_once(max_output_tokens)
        try:
            output_text = self._extract_text(response)
        except RuntimeError as exc:
            if "Could not extract text payload from Responses API response" not in str(exc):
                raise
            logger.warning(
                "openai_extract_text_retry schema=%s reason=no_text_payload",
                schema_name,
            )
            retry_response = _request_once(max_output_tokens)
            output_text = self._extract_text(retry_response)
        try:
            return json.loads(output_text)
        except json.JSONDecodeError as exc:
            retry_tokens = self._retry_tokens_for_truncated_json(max_output_tokens)
            allow_retry_without_cap = max_output_tokens is None
            if (retry_tokens is not None or allow_retry_without_cap) and self._looks_like_truncated_json(
                output_text, exc
            ):
                logger.warning(
                    "openai_json_decode_retry schema=%s reason=likely_truncated_json retry_tokens=%s",
                    schema_name,
                    retry_tokens,
                )
                response_retry = _request_once(retry_tokens)
                retry_text = self._extract_text(response_retry)
                try:
                    return json.loads(retry_text)
                except json.JSONDecodeError as retry_exc:
                    raise RuntimeError(f"Model returned non-JSON output: {retry_text}") from retry_exc
            raise RuntimeError(f"Model returned non-JSON output: {output_text}") from exc

    def text_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        model_override: str | None = None,
        schema_name: str = "text_completion",
    ) -> str:
        if not self._client:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide it to run LLM extraction and planning."
            )
        request_model = (model_override or "").strip() or self.model
        call_attempt = 0

        def _request_once(tokens: int | None) -> Any:
            nonlocal call_attempt
            call_attempt += 1
            started = time.perf_counter()
            try:
                logger.info(
                    "openai_request_start schema=%s model=%s max_output_tokens=%s",
                    schema_name,
                    request_model,
                    tokens,
                )
                response_obj = self._client.responses.create(
                    model=request_model,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                    ],
                    max_output_tokens=tokens,
                )
                self._stats["responses_calls"] = int(self._stats["responses_calls"]) + 1
                duration_ms = int((time.perf_counter() - started) * 1000)
                self._log_call_attempt(
                    schema_name=schema_name,
                    attempt=call_attempt,
                    max_output_tokens=tokens,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    status="ok",
                    duration_ms=duration_ms,
                    model=request_model,
                    response_text=self._extract_text_for_transcript(response_obj),
                )
                return response_obj
            except Exception as exc:
                self._stats["responses_errors"] = int(self._stats["responses_errors"]) + 1
                logger.exception("openai_request_error schema=%s", schema_name)
                duration_ms = int((time.perf_counter() - started) * 1000)
                error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                self._log_call_attempt(
                    schema_name=schema_name,
                    attempt=call_attempt,
                    max_output_tokens=tokens,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    status="error",
                    duration_ms=duration_ms,
                    model=request_model,
                    error_text=error_text,
                )
                if self._looks_like_connection_error(exc):
                    self._disabled_reason = "connection_error"
                    self._client = None
                    logger.warning("openai_disabled reason=%s", self._disabled_reason)
                    self._log_event("openai_client_disabled", "reason=connection_error")
                raise

        response = _request_once(max_output_tokens)
        try:
            return self._extract_text(response)
        except RuntimeError as exc:
            if "Could not extract text payload from Responses API response" not in str(exc):
                raise
            retry_response = _request_once(max_output_tokens)
            return self._extract_text(retry_response)

    def stats(self) -> Dict[str, int]:
        payload = {
            "responses_calls": int(self._stats.get("responses_calls", 0)),
            "responses_errors": int(self._stats.get("responses_errors", 0)),
        }
        if self._disabled_reason:
            payload["disabled_reason"] = self._disabled_reason
        return payload

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

        def _get(node: Any, key: str, default: Any = None) -> Any:
            if node is None:
                return default
            value = getattr(node, key, default)
            if value is not default:
                return value
            if isinstance(node, dict):
                return node.get(key, default)
            return default

        for item in output:
            item_type = str(_get(item, "type", "")).strip().lower()
            if item_type and item_type not in {"message", "output_text", "text"}:
                # Skip non-message items like reasoning traces.
                continue

            content = _get(item, "content", []) or []
            for block in content:
                block_type = str(_get(block, "type", "")).strip().lower()
                if block_type in {"output_text", "text"}:
                    text = _get(block, "text", None)
                    if text:
                        return text
                if block_type in {"output_json", "json"}:
                    payload = _get(block, "json", None)
                    if payload is not None:
                        return json.dumps(payload)

        output_parsed = getattr(response, "output_parsed", None)
        if output_parsed is not None:
            return json.dumps(output_parsed)

        raise RuntimeError("Could not extract text payload from Responses API response")

    @staticmethod
    def _looks_like_connection_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in [
                "connection error",
                "name or service not known",
                "nodename nor servname",
                "temporarily unavailable",
                "timed out",
            ]
        )

    @staticmethod
    def _looks_like_truncated_json(output_text: str, exc: Exception) -> bool:
        message = str(exc).lower()
        stripped = output_text.rstrip()
        if "unterminated string" in message:
            return True
        if "expecting value" in message and stripped.endswith(("{", "[", ":", ",")):
            return True
        if not stripped:
            return False
        if stripped[-1] not in {"}", "]"}:
            return True
        return stripped.count("{") > stripped.count("}") or stripped.count("[") > stripped.count("]")

    @staticmethod
    def _retry_tokens_for_truncated_json(current_tokens: int | None) -> int | None:
        if current_tokens is None:
            return None
        try:
            value = max(1, int(current_tokens))
        except (TypeError, ValueError):
            return None
        if value >= 12_000:
            return None
        return min(12_000, int(value * 1.8) + 200)

    def _log_event(self, title: str, details: str) -> None:
        if not self._transcript_writer:
            return
        try:
            self._transcript_writer.log_event(title=title, details=details)
        except Exception:
            logger.exception("api_transcript_log_event_failed title=%s", title)

    def _log_call_attempt(
        self,
        *,
        schema_name: str,
        attempt: int,
        max_output_tokens: int | None,
        system_prompt: str,
        user_prompt: str,
        status: str,
        duration_ms: int,
        model: str | None = None,
        response_text: str | None = None,
        error_text: str | None = None,
    ) -> None:
        if not self._transcript_writer:
            return
        try:
            self._transcript_writer.log_call(
                schema_name=schema_name,
                model=(model or self.model),
                attempt=attempt,
                max_output_tokens=max_output_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                status=status,
                duration_ms=duration_ms,
                response_text=response_text,
                error_text=error_text,
            )
        except Exception:
            logger.exception("api_transcript_log_call_failed schema=%s attempt=%s", schema_name, attempt)

    def _extract_text_for_transcript(self, response: Any) -> str:
        try:
            return self._extract_text(response)
        except Exception:
            pass
        dump = getattr(response, "model_dump", None)
        if callable(dump):
            try:
                return json.dumps(dump(mode="json"), indent=2)
            except Exception:
                pass
        return repr(response)


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
