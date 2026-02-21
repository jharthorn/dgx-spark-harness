import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "phase60_baseline_manifest_semantic.py"
    spec = importlib.util.spec_from_file_location("phase60_baseline_manifest_semantic", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class Phase60BaselineManifestSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def _context(self) -> dict:
        return {
            "scenario": "rehydrate_replay",
            "model_profile": "llama31_8b_fp8",
            "sweep_replay_concurrencies": [1, 2, 4, 8],
            "baseline_replay_concurrency": 1,
            "pressure_populate_concurrency": 2,
            "pressure_thrash_concurrency": 4,
            "include_b0": False,
            "run_order_per_concurrency": ["B2", "B1"],
            "require_b2_rehydrate": True,
            "io_attrib_enabled": False,
        }

    def _build_run_dir(self, root: Path, run_name: str, *, ts: str, cache_dir: str, results_root: str) -> Path:
        run_dir = root / run_name
        _write_json(
            run_dir / "config.json",
            {
                "run_id": f"run_{ts}",
                "created_utc": f"{ts}:00Z",
                "model_id": "nvidia/Llama-3.1-8B-Instruct-FP8",
                "scenario": "rehydrate_replay",
                "tier_mode": "B2",
                "kv_mode": {
                    "mode": "cpu_disk",
                    "cpu_cache_gb": 2.0,
                    "disk_cache_gb": 32.0,
                    "diagnostic": {"disable_disk_offload_filter": False},
                },
                "args": {
                    "scenario": "rehydrate_replay",
                    "tier_mode": "B2",
                    "kv_mode": "cpu_disk",
                    "kv_cpu_cache_gb": 2.0,
                    "kv_disk_cache_gb": 32.0,
                    "rehydrate_populate_sessions": 16,
                    "rehydrate_thrash_sessions": 192,
                    "rehydrate_turns": 2,
                    "rehydrate_prefix_target_tokens": 4096,
                    "rehydrate_populate_concurrency": 2,
                    "rehydrate_thrash_concurrency": 4,
                    "rehydrate_replay_concurrency": 1,
                    "rehydrate_replay_repeats": 2,
                    "rehydrate_gen_tokens": 128,
                    "seed": 20260210,
                    "request_seed": 20260210,
                    "diagnostic_disable_disk_offload_filter": False,
                    "results_root": results_root,
                    "kvbm_cache_dir": cache_dir,
                },
                "phases": [
                    {"name": "populate", "concurrency": 2, "requests": 32},
                    {"name": "thrash", "concurrency": 4, "requests": 192},
                    {"name": "replay", "concurrency": 1, "requests": 32},
                    {"name": "replay_2", "concurrency": 1, "requests": 32},
                ],
            },
        )
        _write_json(run_dir.parent / "analysis" / "manifest.json", {"scenario": "rehydrate_replay"})
        _write_json(
            run_dir.parent / "analysis" / "worker_runtime_manifest.json",
            {
                "env": {
                    "DYN_KVBM_CPU_CACHE_GB": "2",
                    "DYN_KVBM_DISK_CACHE_GB": "32",
                    "DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER": "0",
                    "HOSTNAME": f"host-{ts}",
                }
            },
        )
        return run_dir

    def test_semantic_hash_stable_across_ts_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = self._build_run_dir(
                root,
                "run_a",
                ts="2026-02-18T23:00:00",
                cache_dir="/mnt/nvme/kvbm/phase60_a",
                results_root="bench/results/phase60_a",
            )
            run_b = self._build_run_dir(
                root,
                "run_b",
                ts="2026-02-19T01:22:33",
                cache_dir="/mnt/nvme/kvbm/phase60_b",
                results_root="bench/results/phase60_b",
            )

            payload_a = self.module.build_semantic_payload(run_a, self._context())
            payload_b = self.module.build_semantic_payload(run_b, self._context())

            self.assertEqual(payload_a["semantic_hash"], payload_b["semantic_hash"])
            self.assertEqual(payload_a["semantic_manifest"], payload_b["semantic_manifest"])

    def test_mismatch_decision_respects_strictness(self) -> None:
        known = {"semantic_hash": "abc", "semantic_manifest": {"scenario": "rehydrate_replay"}}
        current = {"semantic_hash": "def", "semantic_manifest": {"scenario": "rehydrate_replay", "tier_mode": "B2"}}

        strict_decision = self.module.evaluate_baseline_hash_mismatch(
            known_payload=known,
            current_payload=current,
            strict=True,
            accept_new=False,
        )
        self.assertEqual(strict_decision.action, "stop")
        self.assertTrue(strict_decision.should_stop)
        self.assertEqual(strict_decision.reason_code, "baseline_manifest_hash_mismatch")

        warn_decision = self.module.evaluate_baseline_hash_mismatch(
            known_payload=known,
            current_payload=current,
            strict=False,
            accept_new=False,
        )
        self.assertEqual(warn_decision.action, "warn")
        self.assertFalse(warn_decision.should_stop)
        self.assertEqual(warn_decision.warning_code, "BASELINE_MANIFEST_HASH_MISMATCH_WARNING")

        accept_decision = self.module.evaluate_baseline_hash_mismatch(
            known_payload=known,
            current_payload=current,
            strict=True,
            accept_new=True,
        )
        self.assertEqual(accept_decision.action, "accept")
        self.assertFalse(accept_decision.should_stop)
        self.assertEqual(accept_decision.warning_code, "BASELINE_MANIFEST_HASH_ACCEPTED")

    def test_accept_new_baseline_writes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_file = root / "phase60_known_good_baseline_manifest_semantic_hash.json"
            audit_jsonl = root / "phase60_baseline_manifest_audit.jsonl"
            known = {
                "semantic_hash": "old_hash",
                "semantic_manifest": {"scenario": "rehydrate_replay", "tier_mode": "B2"},
                "run_path": "bench/results/known/run",
            }
            current = {
                "semantic_hash": "new_hash",
                "semantic_manifest": {"scenario": "rehydrate_replay", "tier_mode": "B2", "workload_shape": {"rehydrate_turns": 2}},
                "run_path": "bench/results/current/run",
                "manifest_path": "bench/results/current/run/config.json",
            }

            result = self.module.accept_new_baseline(
                baseline_file=baseline_file,
                audit_jsonl=audit_jsonl,
                known_payload=known,
                current_payload=current,
                reason="unit_test_accept",
            )

            self.assertEqual(result["previous_semantic_hash"], "old_hash")
            self.assertEqual(result["new_semantic_hash"], "new_hash")
            self.assertTrue(baseline_file.exists())
            self.assertTrue(audit_jsonl.exists())

            baseline_payload = json.loads(baseline_file.read_text(encoding="utf-8"))
            self.assertEqual(baseline_payload.get("semantic_hash"), "new_hash")
            self.assertEqual(baseline_payload.get("source_run_path"), "bench/results/current/run")

            audit_lines = [line for line in audit_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(audit_lines), 1)
            audit_entry = json.loads(audit_lines[0])
            self.assertEqual(audit_entry.get("previous_semantic_hash"), "old_hash")
            self.assertEqual(audit_entry.get("new_semantic_hash"), "new_hash")
            self.assertTrue(audit_entry.get("changed_fields"))

    def test_boolean_like_strings_normalize_stably(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._build_run_dir(
                root,
                "run_bool",
                ts="2026-02-19T03:14:15",
                cache_dir="/mnt/nvme/kvbm/phase60_bool",
                results_root="bench/results/phase60_bool",
            )
            config_path = run / "config.json"
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            payload["args"]["diagnostic_disable_disk_offload_filter"] = "0"
            config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            context = self._context()
            context["include_b0"] = "0"
            context["require_b2_rehydrate"] = "1"
            context["io_attrib_enabled"] = "false"
            semantic_payload = self.module.build_semantic_payload(run, context)
            manifest = semantic_payload["semantic_manifest"]

            self.assertFalse(manifest["kv_mode"]["diagnostic_disable_disk_offload_filter"])
            self.assertFalse(manifest["sweep_policy"]["include_b0"])
            self.assertTrue(manifest["gates"]["require_b2_rehydrate"])
            self.assertFalse(manifest["gates"]["io_attrib_enabled"])


if __name__ == "__main__":
    unittest.main()
