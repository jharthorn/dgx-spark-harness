"""Thin async client for OpenAI-compatible `/v1/models` and `/v1/completions`."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import httpx

LOG = logging.getLogger(__name__)


@dataclass
class CompletionResponse:
    status_code: int
    latency_ms: float
    request_id: Optional[str]
    text: str
    error: Optional[str]
    ttft_ms: Optional[float]
    ttfc_ms: Optional[float]
    ttfb_ms: Optional[float]
    stream_first_event_type: Optional[str]
    stream_error: Optional[str]
    response_id: Optional[str]
    response_headers: dict[str, str]


class OpenAICompatClient:
    def __init__(self, base_url: str, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        timeout = httpx.Timeout(timeout_s)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def __aenter__(self) -> "OpenAICompatClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_first_model_id(self) -> str:
        payload = await self.fetch_models_payload()
        data = payload.get("data", [])
        if not data:
            raise RuntimeError("No models returned by /v1/models.")
        model_id = data[0].get("id")
        if not model_id:
            raise RuntimeError("Model entry missing `id`.")
        return str(model_id)

    async def fetch_models_payload(self) -> dict[str, Any]:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict):
            raise RuntimeError("Unexpected /v1/models payload.")
        return body

    async def count_models(self) -> int:
        payload = await self.fetch_models_payload()
        data = payload.get("data")
        if isinstance(data, list):
            return len(data)
        return 0

    async def create_completion(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        stop: Optional[Sequence[str]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
        stream_timeout_s: Optional[float] = None,
        stream_record_ttfb: bool = False,
    ) -> CompletionResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)

        if stream:
            return await self._create_completion_streaming(
                payload,
                stream_timeout_s=stream_timeout_s,
                stream_record_ttfb=stream_record_ttfb,
            )
        return await self._create_completion_json(payload, stream_record_ttfb=stream_record_ttfb)

    async def _create_completion_json(
        self, payload: dict[str, Any], *, stream_record_ttfb: bool
    ) -> CompletionResponse:
        start = time.perf_counter()
        status_code = 0
        request_id: Optional[str] = None
        response_headers: dict[str, str] = {}
        ttft_ms: Optional[float] = None
        ttfb_ms: Optional[float] = None
        response_id: Optional[str] = None
        text = ""
        error: Optional[str] = None
        try:
            body_text = ""
            async with self._client.stream("POST", "/v1/completions", json=payload) as resp:
                status_code = resp.status_code
                request_id = resp.headers.get("x-request-id")
                response_headers = _extract_response_header_hints(resp.headers)
                # For non-stream responses this is a first-byte proxy for TTFT.
                header_ms = (time.perf_counter() - start) * 1000.0
                if stream_record_ttfb:
                    ttfb_ms = header_ms
                chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start) * 1000.0
                        chunks.append(chunk)
                if ttft_ms is None:
                    ttft_ms = header_ms
                body_text = b"".join(chunks).decode("utf-8", errors="replace")
            latency_ms = (time.perf_counter() - start) * 1000.0
            try:
                body = json.loads(body_text) if body_text.strip() else {}
                response_id = body.get("id")
                text = _extract_completion_text(body)
                if status_code >= 400:
                    error = _extract_error(body) or f"HTTP {status_code}"
            except Exception:
                if status_code >= 400:
                    error = f"HTTP {status_code}: {body_text[:400]}"
                text = body_text
            return CompletionResponse(
                status_code=status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                text=text,
                error=error,
                ttft_ms=ttft_ms,
                ttfc_ms=None,
                ttfb_ms=ttfb_ms,
                stream_first_event_type=None,
                stream_error=None,
                response_id=response_id,
                response_headers=response_headers,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            if ttft_ms is None:
                ttft_ms = latency_ms
            return CompletionResponse(
                status_code=status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                text=text,
                error=str(exc),
                ttft_ms=ttft_ms,
                ttfc_ms=None,
                ttfb_ms=ttfb_ms,
                stream_first_event_type=None,
                stream_error=None,
                response_id=response_id,
                response_headers=response_headers,
            )

    async def _create_completion_streaming(
        self,
        payload: dict[str, Any],
        *,
        stream_timeout_s: Optional[float],
        stream_record_ttfb: bool,
    ) -> CompletionResponse:
        start = time.perf_counter()
        raw_chunks: list[str] = []
        event_payloads: list[str] = []
        first_chunk_ms: Optional[float] = None
        first_event_ms: Optional[float] = None
        stream_first_event_type: Optional[str] = None
        stream_error: Optional[str] = None
        ttfb_ms: Optional[float] = None
        request_id: Optional[str] = None
        response_id: Optional[str] = None
        status_code = 0
        response_headers: dict[str, str] = {}
        sse_pending = ""
        try:
            timeout = float(stream_timeout_s) if stream_timeout_s is not None else None
            async with self._client.stream("POST", "/v1/completions", json=payload, timeout=timeout) as resp:
                status_code = resp.status_code
                request_id = resp.headers.get("x-request-id")
                response_headers = _extract_response_header_hints(resp.headers)
                header_ms = (time.perf_counter() - start) * 1000.0
                if stream_record_ttfb:
                    ttfb_ms = header_ms
                async for chunk in resp.aiter_text():
                    if chunk:
                        raw_chunks.append(chunk)
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - start) * 1000.0
                        events, sse_pending = _consume_sse_events(sse_pending, chunk)
                        for event in events:
                            payload_text = str(event.get("data") or "")
                            payload_trimmed = payload_text.strip()
                            if not payload_trimmed:
                                continue
                            if first_event_ms is None:
                                first_event_ms = (time.perf_counter() - start) * 1000.0
                                stream_first_event_type = _infer_stream_event_type(payload_trimmed, event.get("event"))
                            if payload_trimmed == "[DONE]":
                                continue
                            event_payloads.append(payload_trimmed)
                            stream_error = stream_error or _extract_stream_payload_error(payload_trimmed)
                for event in _flush_sse_events(sse_pending):
                    payload_text = str(event.get("data") or "")
                    payload_trimmed = payload_text.strip()
                    if not payload_trimmed:
                        continue
                    if first_event_ms is None:
                        first_event_ms = (time.perf_counter() - start) * 1000.0
                        stream_first_event_type = _infer_stream_event_type(payload_trimmed, event.get("event"))
                    if payload_trimmed == "[DONE]":
                        continue
                    event_payloads.append(payload_trimmed)
                    stream_error = stream_error or _extract_stream_payload_error(payload_trimmed)
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw = "".join(raw_chunks)
            parse_source = "\n".join(event_payloads) if event_payloads else raw
            text = _extract_stream_text(parse_source)
            parsed_response_id = _extract_stream_response_id(parse_source)
            if parsed_response_id:
                response_id = parsed_response_id
            elif raw:
                parsed_response_id = _extract_stream_response_id(raw)
                if parsed_response_id:
                    response_id = parsed_response_id
            ttfc_ms = first_event_ms or first_chunk_ms
            if ttfc_ms is None and ttfb_ms is not None:
                ttfc_ms = ttfb_ms
                if stream_first_event_type is None:
                    stream_first_event_type = "headers"
            if ttfc_ms is None:
                ttfc_ms = latency_ms
                if stream_first_event_type is None:
                    stream_first_event_type = "response_end"
            if stream_first_event_type is None and first_chunk_ms is not None:
                stream_first_event_type = "chunk"
            error = stream_error
            if status_code >= 400:
                error = error or f"HTTP {status_code}: {raw[:400]}"
            return CompletionResponse(
                status_code=status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                text=text,
                error=error,
                ttft_ms=ttfc_ms,
                ttfc_ms=ttfc_ms,
                ttfb_ms=ttfb_ms,
                stream_first_event_type=stream_first_event_type,
                stream_error=stream_error,
                response_id=response_id,
                response_headers=response_headers,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            ttfc_ms = first_event_ms or first_chunk_ms
            if ttfc_ms is None and ttfb_ms is not None:
                ttfc_ms = ttfb_ms
            return CompletionResponse(
                status_code=status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                text="",
                error=str(exc),
                ttft_ms=ttfc_ms,
                ttfc_ms=ttfc_ms,
                ttfb_ms=ttfb_ms,
                stream_first_event_type=stream_first_event_type,
                stream_error=(stream_error or str(exc)),
                response_id=response_id,
                response_headers=response_headers,
            )


def _extract_completion_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text
    return ""


def _extract_error(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str):
            return msg
    if isinstance(err, str):
        return err
    return None


def _extract_stream_text(raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    text_parts: list[str] = []
    json_candidates: list[dict[str, Any]] = []
    for line in lines:
        payload = line
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        json_candidates.append(obj)
        chunk_text = _extract_completion_text(obj)
        if chunk_text:
            text_parts.append(chunk_text)
            continue
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta", {})
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        text_parts.append(content)
    if text_parts:
        return "".join(text_parts)
    if json_candidates:
        return _extract_completion_text(json_candidates[-1])
    return raw


def _extract_stream_response_id(raw: str) -> Optional[str]:
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("data:"):
            s = s[len("data:") :].strip()
        if s == "[DONE]":
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            response_id = obj.get("id")
            if isinstance(response_id, str):
                return response_id
    return None


def _extract_stream_payload_error(payload: str) -> Optional[str]:
    try:
        obj = json.loads(payload)
    except Exception:
        return None
    return _extract_error(obj)


def _infer_stream_event_type(payload: str, event_name: Optional[str]) -> str:
    explicit = str(event_name or "").strip().lower()
    if explicit:
        return explicit
    try:
        obj = json.loads(payload)
    except Exception:
        return "data"
    if not isinstance(obj, dict):
        return "data"
    if obj.get("error") is not None:
        return "error"
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str) and delta.get("content"):
                    return "delta_content"
                return "delta"
            if isinstance(first.get("text"), str):
                return "token"
            if first.get("finish_reason") is not None:
                return "finish"
    obj_type = obj.get("object")
    if isinstance(obj_type, str) and obj_type:
        return obj_type
    return "data"


def _consume_sse_events(pending: str, chunk: str) -> tuple[list[dict[str, Optional[str]]], str]:
    buf = f"{pending}{chunk}"
    events: list[dict[str, Optional[str]]] = []
    while True:
        match = re.search(r"\r?\n\r?\n", buf)
        if not match:
            break
        raw_event = buf[: match.start()]
        buf = buf[match.end() :]
        parsed = _parse_sse_event(raw_event)
        if parsed is not None:
            events.append(parsed)
    return events, buf


def _flush_sse_events(pending: str) -> list[dict[str, Optional[str]]]:
    parsed = _parse_sse_event(pending)
    return [parsed] if parsed is not None else []


def _parse_sse_event(raw_event: str) -> Optional[dict[str, Optional[str]]]:
    if not raw_event:
        return None
    event_name: Optional[str] = None
    data_lines: list[str] = []
    for raw_line in raw_event.splitlines():
        line = raw_line.strip("\r")
        if not line:
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field, value = line.split(":", 1)
            value = value.lstrip(" ")
        else:
            field, value = line, ""
        field = field.strip().lower()
        if field == "event":
            event_name = value.strip() or None
        elif field == "data":
            data_lines.append(value)
    if not data_lines:
        return None
    return {"event": event_name, "data": "\n".join(data_lines)}


def _collect_sse_events_from_fragments(fragments: Sequence[str]) -> list[dict[str, Optional[str]]]:
    pending = ""
    out: list[dict[str, Optional[str]]] = []
    for fragment in fragments:
        events, pending = _consume_sse_events(pending, fragment)
        out.extend(events)
    out.extend(_flush_sse_events(pending))
    return out


def _extract_response_header_hints(headers: Any) -> dict[str, str]:
    hints: dict[str, str] = {}
    try:
        for key, value in headers.items():
            lk = str(key).lower()
            if lk.startswith("x-") or "request" in lk or "trace" in lk:
                hints[lk] = str(value)
    except Exception:
        return {}
    return hints
