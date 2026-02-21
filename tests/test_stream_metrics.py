import unittest
import types
import sys
import argparse
import asyncio
import itertools
import tempfile
from pathlib import Path

if "httpx" not in sys.modules:
    sys.modules["httpx"] = types.SimpleNamespace(Timeout=object, AsyncClient=object)

from bench import openai_compat
from bench import run_bench
from bench.prompts import PromptSpec


class TestStreamingMetrics(unittest.TestCase):
    def test_sse_parser_handles_fragmented_chunks(self) -> None:
        fragments = [
            "data: {\"id\":\"resp_1\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n",
            "\n",
            "data: {\"id\":\"resp_1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: [DONE]\n",
            "\n",
        ]
        events = openai_compat._collect_sse_events_from_fragments(fragments)
        self.assertEqual(len(events), 3)
        self.assertIn("\"role\":\"assistant\"", str(events[0].get("data")))
        self.assertIn("\"content\":\"Hello\"", str(events[1].get("data")))
        self.assertEqual(str(events[2].get("data")).strip(), "[DONE]")
        event_type = openai_compat._infer_stream_event_type(str(events[1].get("data")), events[1].get("event"))
        self.assertEqual(event_type, "delta_content")

    def test_summarize_phase_includes_ttfc(self) -> None:
        rows = [
            {
                "latency_ms": 100.0,
                "ttft_ms": 30.0,
                "ttfc_ms": 28.0,
                "ttfb_ms": 10.0,
                "prompt_tokens_est": 256,
                "output_tokens_est": 64,
                "status_code": 200,
                "error": None,
            },
            {
                "latency_ms": 120.0,
                "ttft_ms": 40.0,
                "ttfc_ms": 36.0,
                "ttfb_ms": 12.0,
                "prompt_tokens_est": 300,
                "output_tokens_est": 70,
                "status_code": 200,
                "error": None,
            },
        ]
        summary = run_bench.summarize_phase(rows, duration_s=2.0)
        self.assertIn("p95", summary.get("ttfc_ms") or {})
        self.assertIn("p95", summary.get("ttfb_ms") or {})
        self.assertEqual((summary.get("ttfc_ms") or {}).get("min"), 28.0)
        self.assertEqual((summary.get("ttfc_ms") or {}).get("max"), 36.0)

    def test_summarize_phase_non_stream_keeps_ttfc_empty(self) -> None:
        rows = [
            {
                "latency_ms": 80.0,
                "ttft_ms": 20.0,
                "ttfc_ms": None,
                "ttfb_ms": None,
                "prompt_tokens_est": 200,
                "output_tokens_est": 50,
                "status_code": 200,
                "error": None,
            }
        ]
        summary = run_bench.summarize_phase(rows, duration_s=1.0)
        self.assertIn("p95", summary.get("ttft_ms") or {})
        self.assertEqual(summary.get("ttfc_ms"), {})
        self.assertEqual(summary.get("ttfb_ms"), {})

    def test_run_phase_records_ttfc_fields_when_stream_enabled(self) -> None:
        class _Estimator:
            def estimate(self, text: str) -> int:
                return max(1, len(text.split()))

        class _Client:
            async def create_completion(self, **_: object) -> object:
                return types.SimpleNamespace(
                    latency_ms=123.4,
                    ttft_ms=18.5,
                    ttfc_ms=18.5,
                    ttfb_ms=7.2,
                    status_code=200,
                    request_id="req_1",
                    response_id="resp_1",
                    text="hello world",
                    error=None,
                    stream_first_event_type="delta_content",
                    stream_error=None,
                    response_headers={},
                )

        prompt = PromptSpec(
            prompt_id="p1",
            prompt_set="short",
            target_tokens=128,
            prompt_tokens_est=64,
            prompt="Explain cache reuse briefly.",
            metadata={},
        )
        phase = run_bench.PhasePlan(name="replay", prompts=[prompt], concurrency=1, include_in_overall=True)
        args = argparse.Namespace(
            temperature=0.0,
            top_p=1.0,
            stop=["<|eot_id|>"],
            stream=True,
            request_seed=1337,
            max_tokens=64,
            scenario="standard",
            rehydrate_gen_tokens=None,
            stream_timeout_s=10.0,
            stream_record_ttfb=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            req_path = Path(tmp) / "requests.jsonl"
            with req_path.open("w", encoding="utf-8") as fp:
                rows, _ = asyncio.run(
                    run_bench.run_phase(
                        phase=phase,
                        client=_Client(),
                        model_id="mock-model",
                        args=args,
                        estimator=_Estimator(),
                        request_counter=itertools.count(1),
                        requests_fp=fp,
                        requests_lock=asyncio.Lock(),
                        responses_dir=None,
                    )
                )
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertTrue(row.get("stream"))
            self.assertIsNotNone(row.get("ttfc_ms"))
            self.assertIn("stream_first_event_type", row)
            self.assertIn("http_status", row)

    def test_ttfc_sanity_evaluation_passes_with_stable_ttfc(self) -> None:
        verdict = run_bench.evaluate_ttfc_sanity(
            short_stream_case={
                "summary": {"ttfc_ms": {"p95": 40.0}, "latency_ms": {"p95": 80.0}},
                "error_count": 0,
            },
            long_stream_case={
                "summary": {"ttfc_ms": {"p95": 42.0}, "latency_ms": {"p95": 140.0}},
                "error_count": 0,
            },
            short_legacy_case={"summary": {"ttft_ms": {"p95": 60.0}}, "error_count": 0},
            long_legacy_case={"summary": {"ttft_ms": {"p95": 110.0}}, "error_count": 0},
            ttfc_ratio_threshold=1.35,
            ttft_ratio_threshold=1.20,
        )
        self.assertTrue(verdict["passed"])
        self.assertTrue((verdict.get("checks") or {}).get("ttfc_stable_across_output_length"))
        self.assertTrue((verdict.get("checks") or {}).get("legacy_ttft_scales_with_output_length"))

    def test_ttfc_sanity_evaluation_fails_when_ttfc_scales(self) -> None:
        verdict = run_bench.evaluate_ttfc_sanity(
            short_stream_case={
                "summary": {"ttfc_ms": {"p95": 30.0}, "latency_ms": {"p95": 70.0}},
                "error_count": 0,
            },
            long_stream_case={
                "summary": {"ttfc_ms": {"p95": 65.0}, "latency_ms": {"p95": 160.0}},
                "error_count": 0,
            },
            short_legacy_case={"summary": {"ttft_ms": {"p95": 60.0}}, "error_count": 0},
            long_legacy_case={"summary": {"ttft_ms": {"p95": 95.0}}, "error_count": 0},
            ttfc_ratio_threshold=1.35,
            ttft_ratio_threshold=1.20,
        )
        self.assertFalse(verdict["passed"])
        self.assertIn("ttfc_scales_with_output_length", verdict["reasons"])


if __name__ == "__main__":
    unittest.main()
