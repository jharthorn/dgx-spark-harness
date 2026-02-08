"""Thin async client for OpenAI-compatible `/v1/models` and `/v1/completions`."""

from __future__ import annotations

import json
import logging
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
    response_id: Optional[str]


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
        stop: Optional[Sequence[str]] = None,
        stream: bool = False,
    ) -> CompletionResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stop:
            payload["stop"] = list(stop)

        if stream:
            return await self._create_completion_streaming(payload)
        return await self._create_completion_json(payload)

    async def _create_completion_json(self, payload: dict[str, Any]) -> CompletionResponse:
        start = time.perf_counter()
        try:
            resp = await self._client.post("/v1/completions", json=payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            request_id = resp.headers.get("x-request-id")
            response_id = None
            text = ""
            error: Optional[str] = None
            try:
                body = resp.json()
                response_id = body.get("id")
                text = _extract_completion_text(body)
                if resp.is_error:
                    error = _extract_error(body) or f"HTTP {resp.status_code}"
            except Exception:
                body_text = resp.text
                if resp.is_error:
                    error = f"HTTP {resp.status_code}: {body_text[:400]}"
                text = body_text
            return CompletionResponse(
                status_code=resp.status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                text=text,
                error=error,
                ttft_ms=None,
                response_id=response_id,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return CompletionResponse(
                status_code=0,
                latency_ms=latency_ms,
                request_id=None,
                text="",
                error=str(exc),
                ttft_ms=None,
                response_id=None,
            )

    async def _create_completion_streaming(self, payload: dict[str, Any]) -> CompletionResponse:
        start = time.perf_counter()
        chunks: list[str] = []
        first_chunk_ms: Optional[float] = None
        request_id: Optional[str] = None
        response_id: Optional[str] = None
        status_code = 0
        try:
            async with self._client.stream("POST", "/v1/completions", json=payload) as resp:
                status_code = resp.status_code
                request_id = resp.headers.get("x-request-id")
                async for chunk in resp.aiter_text():
                    if chunk:
                        if first_chunk_ms is None:
                            first_chunk_ms = (time.perf_counter() - start) * 1000.0
                        chunks.append(chunk)
            latency_ms = (time.perf_counter() - start) * 1000.0
            raw = "".join(chunks)
            text = _extract_stream_text(raw)
            parsed_response_id = _extract_stream_response_id(raw)
            if parsed_response_id:
                response_id = parsed_response_id
            error = None
            if status_code >= 400:
                error = f"HTTP {status_code}: {raw[:400]}"
            return CompletionResponse(
                status_code=status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                text=text,
                error=error,
                ttft_ms=first_chunk_ms,
                response_id=response_id,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return CompletionResponse(
                status_code=status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                text="",
                error=str(exc),
                ttft_ms=first_chunk_ms,
                response_id=response_id,
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
