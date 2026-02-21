#!/usr/bin/env bash
set -euo pipefail

normalize_bool() {
  local raw="${1:-0}"
  case "${raw,,}" in
    1|true|yes|on) echo "1" ;;
    0|false|no|off|"") echo "0" ;;
    *)
      echo "Invalid boolean value: ${raw}" >&2
      exit 1
      ;;
  esac
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_ROOT="${BENCH_RESULTS_ROOT:-bench/results}"
TS="${BENCH_PHASE60_TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${RESULTS_ROOT}/phase60_minimal_sweep_logs_${TS}"
mkdir -p "${LOG_DIR}"

CONTAINER_NAME="${BENCH_CONTAINER_NAME:-dyn}"
PYTHON_BIN="${BENCH_PHASE60_PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c "import json" >/dev/null 2>&1; then
  if [[ -x "${REPO_ROOT}/.venv-bench/bin/python3" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv-bench/bin/python3"
  fi
fi

MODEL_PROFILE="${BENCH_MODEL_PROFILE:-llama31_8b_fp8}"
SEED="${BENCH_PHASE60_SEED:-20260210}"
REQUEST_SEED="${BENCH_PHASE60_REQUEST_SEED:-20260210}"
SCENARIO="${BENCH_PHASE60_SCENARIO:-rehydrate_replay}"
CPU_CACHE_GB="${BENCH_PHASE60_CPU_CACHE_GB:-2}"
DISK_CACHE_GB="${BENCH_PHASE60_DISK_CACHE_GB:-32}"
B1_DISK_CACHE_GB="${BENCH_PHASE60_B1_DISK_CACHE_GB:-0}"
B1_KVBM_CACHE_DIR="${BENCH_PHASE60_B1_KVBM_CACHE_DIR:-/tmp/phase60_b1_disk_tier_disabled_${TS}}"
B1_DISK_TIER_READ_BYTES_THRESHOLD="${BENCH_PHASE60_B1_DISK_TIER_READ_BYTES_THRESHOLD:-1048576}"
B1_DISK_TIER_HIT_RATE_THRESHOLD="${BENCH_PHASE60_B1_DISK_TIER_HIT_RATE_THRESHOLD:-0.000001}"
ENFORCE_B1_DISK_TIER_OFF="${BENCH_PHASE60_ENFORCE_B1_DISK_TIER_OFF:-1}"
MAX_TOKENS="${BENCH_PHASE60_MAX_TOKENS:-128}"
TEMPERATURE="${BENCH_PHASE60_TEMPERATURE:-0.2}"
POP_SESSIONS="${BENCH_PHASE60_REHYDRATE_POPULATE_SESSIONS:-16}"
THRASH_SESSIONS="${BENCH_PHASE60_REHYDRATE_THRASH_SESSIONS:-192}"
TURNS="${BENCH_PHASE60_REHYDRATE_TURNS:-2}"
PREFIX_TOKENS="${BENCH_PHASE60_REHYDRATE_PREFIX_TARGET_TOKENS:-4096}"
REPLAY_REPEATS="${BENCH_PHASE60_REHYDRATE_REPLAY_REPEATS:-2}"
REHYDRATE_GEN_TOKENS="${BENCH_PHASE60_REHYDRATE_GEN_TOKENS:-128}"
CONC_LIST="${BENCH_PHASE60_SWEEP_CONCURRENCIES:-1 2 4}"
KVBM_CACHE_BASE_DIR="${BENCH_KVBM_CACHE_BASE_DIR:-/mnt/nvme/kvbm/phase60_minimal_${TS}}"
PRESSURE_POPULATE_CONC="${BENCH_PHASE60_PRESSURE_POPULATE_CONCURRENCY:-2}"
PRESSURE_THRASH_CONC="${BENCH_PHASE60_PRESSURE_THRASH_CONCURRENCY:-2}"
BASELINE_REPLAY_CONC="${BENCH_PHASE60_BASELINE_REPLAY_CONCURRENCY:-1}"
IO_ATTRIB="${BENCH_PHASE60_IO_ATTRIB:-0}"
IO_ATTRIB_INTERVAL_S="${BENCH_PHASE60_IO_ATTRIB_INTERVAL_S:-1}"
STREAM_METRICS="$(normalize_bool "${BENCH_PHASE60_STREAM_METRICS:-1}")"
STREAM_TIMEOUT_S="${BENCH_PHASE60_STREAM_TIMEOUT_S:-}"
STREAM_RECORD_TTFB="$(normalize_bool "${BENCH_PHASE60_STREAM_RECORD_TTFB:-0}")"
REQUIRE_TTFC="$(normalize_bool "${BENCH_PHASE60_REQUIRE_TTFC:-1}")"
TTFC_SANITY_VALIDATE="$(normalize_bool "${BENCH_PHASE60_TTFC_SANITY_VALIDATE:-0}")"
TTFC_SANITY_FAIL_ON_ERROR="$(normalize_bool "${BENCH_PHASE60_TTFC_SANITY_FAIL_ON_ERROR:-0}")"
TTFC_SANITY_REQUESTS="${BENCH_PHASE60_TTFC_SANITY_REQUESTS:-8}"
TTFC_SANITY_CONCURRENCY="${BENCH_PHASE60_TTFC_SANITY_CONCURRENCY:-}"
TTFC_SANITY_SHORT_MAX_TOKENS="${BENCH_PHASE60_TTFC_SANITY_SHORT_MAX_TOKENS:-8}"
TTFC_SANITY_LONG_MAX_TOKENS="${BENCH_PHASE60_TTFC_SANITY_LONG_MAX_TOKENS:-256}"
TTFC_SANITY_TTFC_RATIO_THRESHOLD="${BENCH_PHASE60_TTFC_SANITY_TTFC_RATIO_THRESHOLD:-1.35}"
TTFC_SANITY_TTFT_RATIO_THRESHOLD="${BENCH_PHASE60_TTFC_SANITY_TTFT_RATIO_THRESHOLD:-1.20}"
IO_ATTRIB_CHECKER="${REPO_ROOT}/scripts/check_io_attrib_replay.py"
INCLUDE_B0="${BENCH_PHASE60_INCLUDE_B0:-0}"
STRICT_BASELINE_HASH="${BENCH_PHASE60_STRICT_BASELINE_HASH:-0}"
ACCEPT_NEW_BASELINE_MANIFEST="${BENCH_PHASE60_ACCEPT_NEW_BASELINE_MANIFEST:-0}"

# Preflight defaults inherit full-shape pressure so mechanism signals are comparable.
PREFLIGHT_POP_SESSIONS="${BENCH_PHASE60_PREFLIGHT_POPULATE_SESSIONS:-${POP_SESSIONS}}"
PREFLIGHT_THRASH_SESSIONS="${BENCH_PHASE60_PREFLIGHT_THRASH_SESSIONS:-${THRASH_SESSIONS}}"
PREFLIGHT_TURNS="${BENCH_PHASE60_PREFLIGHT_TURNS:-${TURNS}}"
PREFLIGHT_PREFIX_TOKENS="${BENCH_PHASE60_PREFLIGHT_PREFIX_TARGET_TOKENS:-${PREFIX_TOKENS}}"
PREFLIGHT_REPLAY_REPEATS="${BENCH_PHASE60_PREFLIGHT_REPLAY_REPEATS:-${REPLAY_REPEATS}}"
PREFLIGHT_GEN_TOKENS="${BENCH_PHASE60_PREFLIGHT_GEN_TOKENS:-${REHYDRATE_GEN_TOKENS}}"

KNOWN_GOOD_B2_RUN="${BENCH_PHASE60_KNOWN_GOOD_B2_RUN:-${RESULTS_ROOT}/phase60_rehydrate_B2_r05_20260210T230915Z/run_B2_r05_20260210T230915Z}"
FAILED_B2_C1_RUN="${BENCH_PHASE60_FAILED_B2_C1_RUN:-${RESULTS_ROOT}/phase60_rehydrate_sweep_B2_c1_20260211T000318Z/run_B2_c1_20260211T000318Z}"
BASELINE_SEMANTIC_HASH_FILE="${BENCH_PHASE60_BASELINE_SEMANTIC_HASH_FILE:-${RESULTS_ROOT}/phase60_known_good_baseline_manifest_semantic_hash.json}"
BASELINE_SEMANTIC_AUDIT_JSONL="${BENCH_PHASE60_BASELINE_MANIFEST_AUDIT_JSONL:-${RESULTS_ROOT}/phase60_baseline_manifest_audit.jsonl}"
PHASE60_SEMANTIC_HASH_HELPER="${REPO_ROOT}/scripts/phase60_baseline_manifest_semantic.py"

DIAG_JSON="${RESULTS_ROOT}/phase60_sweep_b2c1_failure_diagnosis_${TS}.json"
SWEEP_SUMMARY_JSON="${RESULTS_ROOT}/phase60_rehydrate_concurrency_sweep_summary_minimal_${TS}.json"
SWEEP_SUMMARY_CSV="${RESULTS_ROOT}/phase60_rehydrate_concurrency_sweep_summary_minimal_${TS}.csv"
STOP_VERDICT_JSON="${RESULTS_ROOT}/phase60_matrix_stop_verdict_minimal_${TS}.json"
RESUME_FROM="${BENCH_PHASE60_RESUME_FROM:-}"
RESUME_SKIP_COMPLETED="${BENCH_PHASE60_RESUME_SKIP_COMPLETED:-true}"
FORCE_NEW_SUMMARY="${BENCH_PHASE60_FORCE_NEW_SUMMARY:-false}"

ensure_container() {
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is not running." >&2
    exit 1
  fi
}

mk_cache_dir() {
  local cache_dir="$1"
  docker exec "${CONTAINER_NAME}" bash -lc "mkdir -p '${cache_dir}' && rm -rf '${cache_dir}'/*"
}

emit_stop_verdict() {
  local reason="$1"
  local detail="$2"
  local mode="${3:-}"
  local conc="${4:-}"
  local diagnostics_json="${5:-}"
  "${PYTHON_BIN}" - "${STOP_VERDICT_JSON}" "${reason}" "${detail}" "${mode}" "${conc}" "${SWEEP_SUMMARY_JSON}" "${diagnostics_json}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
reason = sys.argv[2]
detail = sys.argv[3]
mode = sys.argv[4] or None
conc_raw = sys.argv[5]
summary_path = sys.argv[6]
diagnostics_raw = sys.argv[7] if len(sys.argv) > 7 else ""
conc = int(conc_raw) if conc_raw else None
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "stopped",
    "decision_grade": False,
    "reason": reason,
    "detail": detail,
    "stopped_at_mode": mode,
    "stopped_at_concurrency": conc,
    "sweep_summary_path": summary_path,
}
if diagnostics_raw:
    try:
        payload["diagnostics"] = json.loads(diagnostics_raw)
    except Exception:
        payload["diagnostics_raw"] = diagnostics_raw
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path)
PY
}

append_row() {
  local row_json="$1"
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${row_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
row = json.loads(sys.argv[2])
obj = json.loads(path.read_text())
obj.setdefault("rows", []).append(row)
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

update_summary_stop() {
  local reason="$1"
  local detail="$2"
  local mode="${3:-}"
  local conc="${4:-}"
  local diagnostics_json="${5:-}"
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${reason}" "${detail}" "${mode}" "${conc}" "${diagnostics_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
reason = sys.argv[2]
detail = sys.argv[3]
mode = sys.argv[4] or None
conc = int(sys.argv[5]) if sys.argv[5] else None
diagnostics_raw = sys.argv[6] if len(sys.argv) > 6 else ""
obj = json.loads(path.read_text())
obj["status"] = "stopped"
obj["decision_grade"] = False
obj["stop_reason"] = reason
obj["stop_detail"] = detail
obj["stopped_at"] = {"mode": mode, "concurrency": conc}
if diagnostics_raw:
    try:
        obj["stop_diagnostics"] = json.loads(diagnostics_raw)
    except Exception:
        obj["stop_diagnostics_raw"] = diagnostics_raw
obj.setdefault("baseline_b2_replay_p95_ms_at_concurrency1", None)
obj.setdefault("slo_replay_p95_ms", None)
obj.setdefault("max_concurrency_meeting_slo", {"B0": None, "B1": None, "B2": None})
metric_counts = {"replay_ttfc_p95_ms": 0, "replay_ttft_p95_ms": 0, "missing": 0}
for row in obj.get("rows", []):
    if not isinstance(row, dict):
        continue
    metric_used = row.get("metric_used") if isinstance(row.get("metric_used"), dict) else {}
    replay_p95_used = metric_used.get("replay_p95")
    if replay_p95_used in metric_counts:
        metric_counts[replay_p95_used] += 1
        continue
    if replay_p95_used == "ttfc_ms":
        metric_counts["replay_ttfc_p95_ms"] += 1
        continue
    if replay_p95_used == "ttft_ms":
        metric_counts["replay_ttft_p95_ms"] += 1
        continue
    replay_ttfc = ((row.get("replay_ttfc_ms") or {}).get("p95"))
    replay_ttft = ((row.get("replay_ttft_ms") or {}).get("p95"))
    if replay_ttfc is not None:
        metric_counts["replay_ttfc_p95_ms"] += 1
    elif replay_ttft is not None:
        metric_counts["replay_ttft_p95_ms"] += 1
    else:
        metric_counts["missing"] += 1
dominant_metric = "mixed"
non_zero = [name for name, count in metric_counts.items() if count > 0]
if len(non_zero) == 1:
    dominant_metric = non_zero[0]
elif not non_zero:
    dominant_metric = "missing"
meta = obj.setdefault("meta", {})
meta["metric_policy"] = {
    "preferred": "replay_ttfc_p95_ms",
    "fallback": "replay_ttft_p95_ms",
    "used_replay_p95": dominant_metric,
    "used_replay_p95_counts": metric_counts,
}
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

INCLUDE_B0_ENABLED="0"
if is_truthy "${INCLUDE_B0}"; then
  INCLUDE_B0_ENABLED="1"
fi

STRICT_BASELINE_HASH_ENABLED="0"
if is_truthy "${STRICT_BASELINE_HASH}"; then
  STRICT_BASELINE_HASH_ENABLED="1"
fi

ACCEPT_NEW_BASELINE_MANIFEST_ENABLED="0"
if is_truthy "${ACCEPT_NEW_BASELINE_MANIFEST}"; then
  ACCEPT_NEW_BASELINE_MANIFEST_ENABLED="1"
fi

ENFORCE_B1_DISK_TIER_OFF_ENABLED="0"
if is_truthy "${ENFORCE_B1_DISK_TIER_OFF}"; then
  ENFORCE_B1_DISK_TIER_OFF_ENABLED="1"
fi

REQUIRE_TTFC_ENABLED="0"
if is_truthy "${REQUIRE_TTFC}"; then
  REQUIRE_TTFC_ENABLED="1"
fi

summary_append_warning() {
  local code="$1"
  local detail="$2"
  local mode="${3:-}"
  local conc="${4:-}"
  local diagnostics_json="${5:-}"
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${code}" "${detail}" "${mode}" "${conc}" "${diagnostics_json}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
code = sys.argv[2]
detail = sys.argv[3]
mode = sys.argv[4] or None
conc = int(sys.argv[5]) if sys.argv[5] else None
diagnostics_raw = sys.argv[6] if len(sys.argv) > 6 else ""

obj = json.loads(path.read_text())
warnings = obj.setdefault("warnings", [])
entry = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "status": "warning",
    "code": code,
    "detail": detail,
    "mode": mode,
    "concurrency": conc,
}
if diagnostics_raw:
    try:
        entry["diagnostics"] = json.loads(diagnostics_raw)
    except Exception:
        entry["diagnostics_raw"] = diagnostics_raw
warnings.append(entry)
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

summary_set_baseline_policy_meta() {
  local known_hash="$1"
  local known_source="$2"
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${STRICT_BASELINE_HASH_ENABLED}" "${ACCEPT_NEW_BASELINE_MANIFEST_ENABLED}" "${BASELINE_SEMANTIC_HASH_FILE}" "${BASELINE_SEMANTIC_AUDIT_JSONL}" "${known_hash}" "${known_source}" "${KNOWN_GOOD_B2_RUN}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
strict_mode = bool(int(sys.argv[2]))
accept_new = bool(int(sys.argv[3]))
baseline_file = sys.argv[4]
audit_jsonl = sys.argv[5]
known_hash = sys.argv[6] or None
known_source = sys.argv[7] or None
known_good_b2_run = sys.argv[8] or None

obj = json.loads(path.read_text())
meta = obj.setdefault("meta", {})
meta["baseline_manifest_hash_policy"] = {
    "strict_mode": strict_mode,
    "accept_new_baseline_manifest": accept_new,
    "baseline_semantic_hash_file": baseline_file,
    "baseline_semantic_audit_jsonl": audit_jsonl,
    "known_good_run_fallback": known_good_b2_run,
    "known_semantic_hash": known_hash,
    "known_semantic_source": known_source,
}
obj.setdefault("warnings", [])
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

summary_upsert_row() {
  local row_json="$1"
  local point_key="${2:-}"
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${row_json}" "${point_key}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
row = json.loads(sys.argv[2])
point_key = sys.argv[3]
obj = json.loads(path.read_text())
rows = obj.setdefault("rows", [])

if point_key:
    row["point_key"] = point_key
    for i, existing in enumerate(rows):
        if existing.get("point_key") == point_key:
            rows[i] = row
            break
    else:
        rows.append(row)
else:
    rows.append(row)

path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

summary_has_status() {
  local point_key="$1"
  local statuses_csv="$2"
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${point_key}" "${statuses_csv}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
point_key = sys.argv[2]
statuses = {s.strip() for s in sys.argv[3].split(",") if s.strip()}
obj = json.loads(path.read_text())
for row in obj.get("rows", []):
    if row.get("point_key") == point_key and row.get("status") in statuses:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

summary_has_baseline() {
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
obj = json.loads(path.read_text())
baseline = obj.get("baseline_b2_replay_p95_ms_at_concurrency1")
if baseline is None:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

summary_set_baseline_from_metrics() {
  local metrics_json="$1"
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${metrics_json}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
metrics = json.loads(sys.argv[2])
baseline = ((metrics.get("replay_latency_ms") or {}).get("p95"))
obj = json.loads(path.read_text())
if baseline is None:
    print("0")
    raise SystemExit(0)
obj["baseline_b2_replay_p95_ms_at_concurrency1"] = baseline
obj["slo_replay_p95_ms"] = float(baseline) * 1.10
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
print("1")
PY
}

collect_run_metrics() {
  local run_dir="$1"
  "${PYTHON_BIN}" - "${run_dir}" "${B1_DISK_TIER_READ_BYTES_THRESHOLD}" "${B1_DISK_TIER_HIT_RATE_THRESHOLD}" "${ENFORCE_B1_DISK_TIER_OFF_ENABLED}" <<'PY'
import hashlib
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
b1_read_bytes_threshold_raw = sys.argv[2]
b1_disk_hit_rate_threshold_raw = sys.argv[3]
enforce_b1_disk_tier_off_raw = sys.argv[4]

def parse_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def first_float(*values):
    for value in values:
        parsed = parse_float(value)
        if parsed is not None:
            return parsed
    return None

summary = json.loads((run_dir / "summary.json").read_text())
config = json.loads((run_dir / "config.json").read_text())
manifest = json.loads((run_dir / "manifest.json").read_text())
runtime_manifest = {}
runtime_manifest_path = run_dir.parent / "analysis" / "worker_runtime_manifest.json"
if runtime_manifest_path.exists():
    try:
        runtime_manifest = json.loads(runtime_manifest_path.read_text())
    except Exception:
        runtime_manifest = {}
runtime_env = runtime_manifest.get("env") if isinstance(runtime_manifest.get("env"), dict) else {}

phases = {p.get("phase"): p for p in summary.get("phase_summaries", []) if p.get("phase")}
overall = summary.get("overall_summary", {})
ttft = overall.get("ttft_ms") or {}
ttfc = overall.get("ttfc_ms") or {}
replay_ttft = (phases.get("replay", {}) or {}).get("ttft_ms") or {}
replay_ttfc = (phases.get("replay", {}) or {}).get("ttfc_ms") or {}
replay_lat = (phases.get("replay", {}) or {}).get("latency_ms") or {}

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

replay_kv = load_json(run_dir / "phase_deltas/phase_replay_kvbm_metrics_delta.json")
replay2_kv_path = run_dir / "phase_deltas/phase_replay_2_kvbm_metrics_delta.json"
replay2_kv = load_json(replay2_kv_path) if replay2_kv_path.exists() else {}
replay_io = load_json(run_dir / "phase_deltas/phase_replay_os_io_delta.json")
replay2_io_path = run_dir / "phase_deltas/phase_replay_2_os_io_delta.json"
replay2_io = load_json(replay2_io_path) if replay2_io_path.exists() else {}

kvbm_status = summary.get("kvbm_metrics_status")
if not isinstance(kvbm_status, dict):
    kvbm_status = ((summary.get("kvbm_metrics") or {}).get("kvbm_metrics_status")) if isinstance(summary.get("kvbm_metrics"), dict) else {}
if not isinstance(kvbm_status, dict):
    kvbm_status = {}
kvbm_ok = str(kvbm_status.get("status") or "") == "ok"

def choose_metric(scope, ttfc_stats, ttft_stats, stat):
    if isinstance(ttfc_stats, dict):
        ttfc_value = parse_float(ttfc_stats.get(stat))
        if ttfc_value is not None:
            return ttfc_value, f"{scope}_ttfc_{stat}_ms"
    if isinstance(ttft_stats, dict):
        ttft_value = parse_float(ttft_stats.get(stat))
        if ttft_value is not None:
            return ttft_value, f"{scope}_ttft_{stat}_ms"
    return None, "missing"

overall_metric_p95, overall_metric_source_p95 = choose_metric("overall", ttfc, ttft, "p95")
overall_metric_p99, overall_metric_source_p99 = choose_metric("overall", ttfc, ttft, "p99")
replay_metric_p95, replay_metric_source_p95 = choose_metric("replay", replay_ttfc, replay_ttft, "p95")
replay_metric_p99, replay_metric_source_p99 = choose_metric("replay", replay_ttfc, replay_ttft, "p99")

phase_io_gib = {}
for phase_name, phase_payload in phases.items():
    if not isinstance(phase_payload, dict):
        continue
    io = phase_payload.get("io_delta") or {}
    read_mib = float(io.get("read_mib_delta") or 0.0)
    write_mib = float(io.get("write_mib_delta") or 0.0)
    phase_io_gib[str(phase_name)] = {
        "read_gib": round(read_mib / 1024.0, 6),
        "write_gib": round(write_mib / 1024.0, 6),
    }

rb = replay_io.get("block_device_delta", {})
rb2 = replay2_io.get("block_device_delta", {})
wb = replay_io.get("worker_process_io_delta", {})
wb2 = replay2_io.get("worker_process_io_delta", {})

matched_delta = (
    (replay_kv.get("kvbm_matched_tokens_delta") or 0.0) + (replay2_kv.get("kvbm_matched_tokens_delta") or 0.0)
    if kvbm_ok
    else None
)
onboard_delta = (
    (replay_kv.get("kvbm_onboard_blocks_d2d_delta") or 0.0) + (replay2_kv.get("kvbm_onboard_blocks_d2d_delta") or 0.0)
    if kvbm_ok
    else None
)
offload_h2d_delta = (
    (replay_kv.get("kvbm_offload_blocks_h2d_delta") or 0.0) + (replay2_kv.get("kvbm_offload_blocks_h2d_delta") or 0.0)
    if kvbm_ok
    else None
)
block_read_delta = (rb.get("read_bytes_delta") or 0) + (rb2.get("read_bytes_delta") or 0)
block_write_delta = (rb.get("write_bytes_delta") or 0) + (rb2.get("write_bytes_delta") or 0)
cgroup_read_delta = (wb.get("cgroup_read_bytes_delta") or 0) + (wb2.get("cgroup_read_bytes_delta") or 0)
cgroup_write_delta = (wb.get("cgroup_write_bytes_delta") or 0) + (wb2.get("cgroup_write_bytes_delta") or 0)

replay_phase = phases.get("replay", {}) if isinstance(phases.get("replay"), dict) else {}
replay_kv_end = replay_phase.get("kvbm_metrics_end") if isinstance(replay_phase.get("kvbm_metrics_end"), dict) else {}
replay_kv_start = replay_phase.get("kvbm_metrics_start") if isinstance(replay_phase.get("kvbm_metrics_start"), dict) else {}
replay_kv_delta = replay_phase.get("kvbm_metrics_delta") if isinstance(replay_phase.get("kvbm_metrics_delta"), dict) else {}
disk_hit_rate = first_float(
    replay_kv_end.get("kvbm_disk_cache_hit_rate"),
    replay_kv_delta.get("kvbm_disk_cache_hit_rate"),
    replay_kv_delta.get("kvbm_disk_cache_hit_rate_delta"),
    replay_kv.get("kvbm_disk_cache_hit_rate"),
    replay2_kv.get("kvbm_disk_cache_hit_rate"),
)
if disk_hit_rate is None:
    end_rate = parse_float(replay_kv_end.get("kvbm_disk_cache_hit_rate"))
    start_rate = parse_float(replay_kv_start.get("kvbm_disk_cache_hit_rate"))
    if end_rate is not None and start_rate is not None:
        disk_hit_rate = end_rate - start_rate
    else:
        disk_hit_rate = 0.0

request_manifest = run_dir / "request_manifest.jsonl"
prompts_manifest = run_dir / "prompts_manifest.jsonl"

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

mode_value = summary.get("mode") or summary.get("tier_mode") or ((config.get("tier_mode")) if isinstance(config, dict) else None)
mode_upper = str(mode_value).upper() if mode_value is not None else ""
config_args = config.get("args") if isinstance(config.get("args"), dict) else {}
config_kv_mode = config.get("kv_mode") if isinstance(config.get("kv_mode"), dict) else {}
kv_mode_name = config_kv_mode.get("mode") or config.get("kv_mode") or config_args.get("kv_mode")

config_disk_cache_gb = first_float(
    config_args.get("kv_disk_cache_gb"),
    config_kv_mode.get("disk_cache_gb"),
    ((manifest.get("kv_mode") or {}).get("disk_cache_gb") if isinstance(manifest.get("kv_mode"), dict) else None),
)
runtime_disk_cache_gb = first_float(runtime_env.get("DYN_KVBM_DISK_CACHE_GB"))
config_cache_dir = config_args.get("kvbm_cache_dir")
runtime_cache_dir = runtime_env.get("DYN_KVBM_DISK_CACHE_DIR")

try:
    b1_read_bytes_threshold = max(0, int(float(b1_read_bytes_threshold_raw)))
except Exception:
    b1_read_bytes_threshold = 0
try:
    b1_disk_hit_rate_threshold = max(0.0, float(b1_disk_hit_rate_threshold_raw))
except Exception:
    b1_disk_hit_rate_threshold = 0.0
enforce_b1_disk_tier_off = parse_bool(enforce_b1_disk_tier_off_raw)

b1_disk_tier_verified = None
b1_disk_tier_verification = None
if mode_upper == "B1":
    fail_reasons = []
    kv_mode_ok = str(kv_mode_name or "") == "cpu_only"
    disk_cache_disabled = (
        (config_disk_cache_gb is not None and config_disk_cache_gb <= 0.0)
        or (runtime_disk_cache_gb is not None and runtime_disk_cache_gb <= 0.0)
    )
    if not kv_mode_ok:
        fail_reasons.append(f"kv_mode_not_cpu_only:{kv_mode_name}")
    if not disk_cache_disabled:
        fail_reasons.append(
            f"disk_cache_gb_not_disabled:config={config_disk_cache_gb},runtime={runtime_disk_cache_gb}"
        )
    if float(onboard_delta or 0.0) > 0.0:
        fail_reasons.append(f"onboard_blocks_d2d_delta={onboard_delta}")
    if int(block_read_delta) > b1_read_bytes_threshold:
        fail_reasons.append(f"replay_block_read_bytes_delta={block_read_delta}")
    if int(cgroup_read_delta) > b1_read_bytes_threshold:
        fail_reasons.append(f"replay_process_read_bytes_delta={cgroup_read_delta}")
    if float(disk_hit_rate or 0.0) > b1_disk_hit_rate_threshold:
        fail_reasons.append(f"kvbm_disk_cache_hit_rate={disk_hit_rate}")

    b1_disk_tier_verified = len(fail_reasons) == 0
    b1_disk_tier_verification = {
        "enforcement_enabled": enforce_b1_disk_tier_off,
        "kv_mode": kv_mode_name,
        "config_disk_cache_gb": config_disk_cache_gb,
        "runtime_disk_cache_gb": runtime_disk_cache_gb,
        "config_cache_dir": config_cache_dir,
        "runtime_cache_dir": runtime_cache_dir,
        "onboard_blocks_d2d_delta_replay_plus_replay2": float(onboard_delta or 0.0),
        "replay_block_read_bytes_delta_replay_plus_replay2": int(block_read_delta),
        "replay_process_read_bytes_delta_replay_plus_replay2": int(cgroup_read_delta),
        "kvbm_disk_cache_hit_rate": float(disk_hit_rate or 0.0),
        "read_bytes_threshold": b1_read_bytes_threshold,
        "disk_hit_rate_threshold": b1_disk_hit_rate_threshold,
        "fail_reasons": fail_reasons,
    }

payload = {
    "mode": mode_value,
    "stream": bool(summary.get("stream")),
    "stream_record_ttfb": bool(summary.get("stream_record_ttfb")),
    "kvbm_enabled": bool(summary.get("kvbm_enabled")) if "kvbm_enabled" in summary else bool(((summary.get("kv_mode") or {}).get("kvbm_enabled"))),
    "kvbm_metrics_available": bool(summary.get("kvbm_metrics_available")) if "kvbm_metrics_available" in summary else bool(((summary.get("kvbm_metrics") or {}).get("available"))),
    "kvbm_metrics_status": kvbm_status,
    "run_path": str(run_dir),
    "error_rate": overall.get("error_rate"),
    "overall_ttfc_ms": {
        "p50": ttfc.get("p50"),
        "p95": ttfc.get("p95"),
        "p99": ttfc.get("p99"),
    },
    "overall_ttft_ms": {
        "p50": ttft.get("p50"),
        "p95": ttft.get("p95"),
        "p99": ttft.get("p99"),
    },
    "replay_ttfc_ms": {
        "p50": replay_ttfc.get("p50"),
        "p95": replay_ttfc.get("p95"),
        "p99": replay_ttfc.get("p99"),
    },
    "replay_ttft_ms": {
        "p50": replay_ttft.get("p50"),
        "p95": replay_ttft.get("p95"),
        "p99": replay_ttft.get("p99"),
    },
    "replay_latency_ms": {
        "p50": replay_lat.get("p50"),
        "p95": replay_lat.get("p95"),
        "p99": replay_lat.get("p99"),
    },
    "metric_preferred": "replay_ttfc_p95_ms",
    "metric_used": {
        "overall_p95": overall_metric_source_p95,
        "overall_p99": overall_metric_source_p99,
        "replay_p95": replay_metric_source_p95,
        "replay_p99": replay_metric_source_p99,
    },
    "overall_metric_p95_ms": overall_metric_p95,
    "overall_metric_p99_ms": overall_metric_p99,
    "replay_metric_p95_ms": replay_metric_p95,
    "replay_metric_p99_ms": replay_metric_p99,
    "b1_disk_tier_verified": b1_disk_tier_verified,
    "b1_disk_tier_verification": b1_disk_tier_verification,
    "mechanism": {
        "kvbm_matched_tokens_delta_replay_plus_replay2": matched_delta,
        "kvbm_onboard_blocks_d2d_delta_replay_plus_replay2": onboard_delta,
        "kvbm_offload_blocks_h2d_delta_replay_plus_replay2": offload_h2d_delta,
        "kvbm_disk_cache_hit_rate": disk_hit_rate if kvbm_ok else None,
        "block_read_bytes_delta_replay_plus_replay2": block_read_delta,
        "block_write_bytes_delta_replay_plus_replay2": block_write_delta,
        "cgroup_read_bytes_delta_replay_plus_replay2": cgroup_read_delta,
        "cgroup_write_bytes_delta_replay_plus_replay2": cgroup_write_delta,
        "ssd_rehydrate_signal_present": bool(
            float(onboard_delta or 0.0) > 0.0
            or float(disk_hit_rate or 0.0) > 0.0
            or int(block_read_delta) > 0
            or int(cgroup_read_delta) > 0
        ),
        "ssd_write_signal_present": bool(
            float(offload_h2d_delta or 0.0) > 0.0
            or int(block_write_delta) > 0
            or int(cgroup_write_delta) > 0
        ),
    },
    "phase_io_gib": phase_io_gib,
    "manifest_hashes": {
        "request_manifest_sha256": sha256(request_manifest),
        "prompts_manifest_sha256": sha256(prompts_manifest),
    },
    "manifest_counts": {
        "request_manifest_rows": sum(1 for _ in request_manifest.open("r", encoding="utf-8")),
        "prompts_manifest_rows": sum(1 for _ in prompts_manifest.open("r", encoding="utf-8")),
    },
    "config_snapshot": {
        "scenario": config.get("scenario"),
        "tier_mode": config.get("tier_mode"),
        "kv_mode": (((config.get("kv_mode") or {}).get("mode")) if isinstance(config.get("kv_mode"), dict) else config.get("kv_mode")),
        "kv_disk_cache_gb": ((config.get("args") or {}).get("kv_disk_cache_gb")),
        "runtime_kv_disk_cache_gb": runtime_env.get("DYN_KVBM_DISK_CACHE_GB"),
        "rehydrate_populate_sessions": ((config.get("args") or {}).get("rehydrate_populate_sessions")),
        "rehydrate_thrash_sessions": ((config.get("args") or {}).get("rehydrate_thrash_sessions")),
        "rehydrate_turns": ((config.get("args") or {}).get("rehydrate_turns")),
        "rehydrate_prefix_target_tokens": ((config.get("args") or {}).get("rehydrate_prefix_target_tokens")),
        "rehydrate_populate_concurrency": ((config.get("args") or {}).get("rehydrate_populate_concurrency")),
        "rehydrate_thrash_concurrency": ((config.get("args") or {}).get("rehydrate_thrash_concurrency")),
        "rehydrate_replay_concurrency": ((config.get("args") or {}).get("rehydrate_replay_concurrency")),
        "rehydrate_replay_repeats": ((config.get("args") or {}).get("rehydrate_replay_repeats")),
        "request_seed": ((config.get("args") or {}).get("request_seed")),
        "seed": ((config.get("args") or {}).get("seed")),
        "kvbm_cache_dir": ((config.get("args") or {}).get("kvbm_cache_dir")),
        "runtime_kvbm_cache_dir": runtime_env.get("DYN_KVBM_DISK_CACHE_DIR"),
        "diagnostic_disable_disk_offload_filter": ((config.get("args") or {}).get("diagnostic_disable_disk_offload_filter")),
    },
    "nvme_artifacts": {},
    "device_metadata_post": None,
    "io_attribution": None,
    "io_attribution_verdict": None,
}

for name in ("nvme_identity.json", "nvme_smart_pre.json", "nvme_smart_post.json"):
    p = run_dir / name
    if p.exists():
        d = json.loads(p.read_text())
        payload["nvme_artifacts"][name] = {
            "success": d.get("success"),
            "payload_ok": d.get("payload_ok"),
            "format": d.get("format"),
        }

device_metadata_post_path = run_dir / "device_metadata_post.json"
if device_metadata_post_path.exists():
    try:
        dmp = json.loads(device_metadata_post_path.read_text())
        dmp_summary = dmp.get("primary_storage_summary") or {}
        dmp_targets = dmp.get("resolved_targets") or {}
        payload["device_metadata_post"] = {
            "available": True,
            "capture_timestamp": dmp.get("capture_timestamp"),
            "capture_error_count": len(dmp.get("capture_errors") or []),
            "primary_nvme_device": dmp.get("primary_nvme_device") or dmp_targets.get("primary_nvme_device"),
            "primary_nvme_namespace": dmp.get("primary_nvme_namespace") or dmp_targets.get("primary_nvme_namespace"),
            "primary_nvme_model": dmp_summary.get("model"),
            "primary_nvme_fw": dmp_summary.get("firmware_rev"),
            "primary_nvme_serial": dmp_summary.get("serial"),
            "primary_nvme_size": dmp_summary.get("size"),
            "pcie_link": dmp_summary.get("pcie_link"),
        }
    except Exception as exc:
        payload["device_metadata_post"] = {
            "available": False,
            "error": str(exc),
        }
else:
    payload["device_metadata_post"] = {
        "available": False,
        "error": "missing_device_metadata_post",
    }

io_report_path = run_dir / "io" / "io_attribution_report.json"
if io_report_path.exists():
    io_rep = json.loads(io_report_path.read_text())
    block_by_phase = io_rep.get("block_io_by_phase") or {}
    proc_by_phase = io_rep.get("process_io_by_phase") or {}
    replay_phase = None
    for candidate in ("replay", "replay_A"):
        if candidate in block_by_phase or candidate in proc_by_phase:
            replay_phase = candidate
            break
    if replay_phase is None:
        replay_like = sorted(
            name
            for name in set(block_by_phase.keys()) | set(proc_by_phase.keys())
            if str(name).startswith("replay")
        )
        replay_phase = replay_like[0] if replay_like else None
    replay_block = ((block_by_phase or {}).get(replay_phase) or {}) if replay_phase else {}
    replay_proc = ((proc_by_phase or {}).get(replay_phase) or {}) if replay_phase else {}
    payload["io_attribution"] = {
        "available": io_rep.get("available"),
        "capture_error_count": len(io_rep.get("capture_errors") or []),
        "kvbm_disk_path": io_rep.get("kvbm_disk_path"),
        "replay_phase": replay_phase,
        "replay_block_read_bytes": replay_block.get("read_bytes"),
        "replay_process_read_bytes": replay_proc.get("read_bytes"),
    }

io_verdict_path = run_dir / "io" / "io_attrib_verdict.json"
if io_verdict_path.exists():
    try:
        verdict = json.loads(io_verdict_path.read_text())
        checks = verdict.get("checks") if isinstance(verdict.get("checks"), list) else []
        failed = [c for c in checks if isinstance(c, dict) and c.get("status") == "FAIL"]
        warns = [c for c in checks if isinstance(c, dict) and c.get("status") == "WARN"]
        payload["io_attribution_verdict"] = {
            "available": True,
            "pass": bool(verdict.get("pass")),
            "mode": verdict.get("mode"),
            "strict_replay_gate_required": bool(verdict.get("strict_replay_gate_required")),
            "replay_phase": verdict.get("replay_phase"),
            "replay_read_bytes_block": verdict.get("replay_read_bytes_block"),
            "replay_read_bytes_process_total": verdict.get("replay_read_bytes_process_total"),
            "top_pids_replay": list(verdict.get("top_pids_replay") or []),
            "fail_count": len(failed),
            "warn_count": len(warns),
            "failed_checks": [str(c.get("name")) for c in failed if c.get("name")],
            "warn_checks": [str(c.get("name")) for c in warns if c.get("name")],
            "checks": checks,
            "timestamp_utc": verdict.get("timestamp_utc"),
            "verdict_path": str(io_verdict_path),
        }
    except Exception as exc:
        payload["io_attribution_verdict"] = {
            "available": False,
            "error": f"failed_to_parse_io_attrib_verdict:{exc}",
            "verdict_path": str(io_verdict_path),
        }
else:
    payload["io_attribution_verdict"] = {
        "available": False,
        "error": "missing_io_attrib_verdict",
        "verdict_path": str(io_verdict_path),
    }

print(json.dumps(payload, separators=(",", ":")))
PY
}

build_semantic_context_json() {
  "${PYTHON_BIN}" - "${SCENARIO}" "${MODEL_PROFILE}" "${CONC_LIST}" "${BASELINE_REPLAY_CONC}" "${PRESSURE_POPULATE_CONC}" "${PRESSURE_THRASH_CONC}" "${IO_ATTRIB}" "${INCLUDE_B0_ENABLED}" "${ENFORCE_B1_DISK_TIER_OFF_ENABLED}" "${B1_DISK_CACHE_GB}" "${B1_DISK_TIER_READ_BYTES_THRESHOLD}" "${B1_DISK_TIER_HIT_RATE_THRESHOLD}" <<'PY'
import json
import sys

scenario = sys.argv[1]
model_profile = sys.argv[2]
conc_list_raw = sys.argv[3]
baseline_replay_conc = int(sys.argv[4])
pressure_pop_conc = int(sys.argv[5])
pressure_thrash_conc = int(sys.argv[6])
io_attrib_enabled = str(sys.argv[7]).strip().lower() in {"1", "true", "yes", "on"}
include_b0 = str(sys.argv[8]).strip().lower() in {"1", "true", "yes", "on"}
enforce_b1_disk_tier_off = str(sys.argv[9]).strip().lower() in {"1", "true", "yes", "on"}
b1_disk_cache_gb = float(sys.argv[10])
b1_read_bytes_threshold = int(float(sys.argv[11]))
b1_disk_hit_rate_threshold = float(sys.argv[12])
sweep_concs = [int(x) for x in str(conc_list_raw).split() if x.strip()]
run_order = ["B2", "B1"] + (["B0"] if include_b0 else [])

payload = {
    "scenario": scenario,
    "model_profile": model_profile,
    "sweep_replay_concurrencies": sweep_concs,
    "baseline_replay_concurrency": baseline_replay_conc,
    "pressure_populate_concurrency": pressure_pop_conc,
    "pressure_thrash_concurrency": pressure_thrash_conc,
    "include_b0": include_b0,
    "run_order_per_concurrency": run_order,
    "require_b2_rehydrate": True,
    "io_attrib_enabled": io_attrib_enabled,
    "b1_disk_tier_enforced": enforce_b1_disk_tier_off,
    "b1_disk_cache_gb": b1_disk_cache_gb,
    "b1_read_bytes_threshold": b1_read_bytes_threshold,
    "b1_disk_hit_rate_threshold": b1_disk_hit_rate_threshold,
    "metric_preferred": "replay_ttfc_p95_ms",
}
print(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
PY
}

compute_semantic_payload_for_run() {
  local run_dir="$1"
  local context_json="$2"
  "${PYTHON_BIN}" "${PHASE60_SEMANTIC_HASH_HELPER}" compute --run-dir "${run_dir}" --context-json "${context_json}"
}

resolve_known_semantic_payload() {
  local baseline_file="$1"
  local fallback_payload_json="$2"
  "${PYTHON_BIN}" - "${baseline_file}" "${fallback_payload_json}" <<'PY'
import json
import pathlib
import sys

baseline_file = pathlib.Path(sys.argv[1])
fallback_payload = json.loads(sys.argv[2])

if baseline_file.exists():
    try:
        payload = json.loads(baseline_file.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, dict) and payload.get("semantic_hash"):
        merged = {
            "semantic_hash": payload.get("semantic_hash"),
            "semantic_manifest": payload.get("semantic_manifest") if isinstance(payload.get("semantic_manifest"), dict) else {},
            "run_path": payload.get("source_run_path"),
            "manifest_path": payload.get("manifest_path"),
            "source": "baseline_file",
            "baseline_file": str(baseline_file),
        }
        print(json.dumps(merged, separators=(",", ":"), ensure_ascii=True))
        raise SystemExit(0)

fallback_payload["source"] = "known_good_run_fallback"
print(json.dumps(fallback_payload, separators=(",", ":"), ensure_ascii=True))
PY
}

run_probe() {
  local mode="$1"
  local bundle_id="$2"
  local run_id="$3"
  local replay_conc="$4"
  local cache_dir="$5"
  local disable_filter="$6"
  local populate_sessions="$7"
  local thrash_sessions="$8"
  local turns="$9"
  local prefix_tokens="${10}"
  local replay_repeats="${11}"
  local gen_tokens="${12}"
  local log_path="${LOG_DIR}/${bundle_id}.log"
  local mode_cache_dir="${cache_dir}"
  local mode_disk_cache_gb="${DISK_CACHE_GB}"

  if [[ "${mode}" == "B1" && "${ENFORCE_B1_DISK_TIER_OFF_ENABLED}" == "1" ]]; then
    mode_disk_cache_gb="${B1_DISK_CACHE_GB}"
    mode_cache_dir="${B1_KVBM_CACHE_DIR}/c${replay_conc}"
  fi
  if [[ "${mode}" == "B0" ]]; then
    mode_disk_cache_gb="0"
    mode_cache_dir=""
  fi

  if [[ "${mode}" != "B0" && -n "${mode_cache_dir}" ]]; then
    mk_cache_dir "${mode_cache_dir}"
  fi
  {
    echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_probe start mode=${mode} bundle=${bundle_id} run=${run_id} replay_conc=${replay_conc} pressure_populate_conc=${PRESSURE_POPULATE_CONC} pressure_thrash_conc=${PRESSURE_THRASH_CONC} cache_dir=${mode_cache_dir} mode_disk_cache_gb=${mode_disk_cache_gb} disable_filter=${disable_filter} ===="
  } >> "${log_path}"

  set +e
  BENCH_RESULTS_ROOT="${RESULTS_ROOT}" \
  BENCH_PHASE56_LIKE_BUNDLE_ID="${bundle_id}" \
  BENCH_PHASE56_LIKE_RUN_ID="${run_id}" \
  BENCH_TIER_MODE="${mode}" \
  BENCH_MODEL_PROFILE="${MODEL_PROFILE}" \
  BENCH_PHASE56_SCENARIO="${SCENARIO}" \
  BENCH_PHASE56_CPU_CACHE_GB="${CPU_CACHE_GB}" \
  BENCH_PHASE56_DISK_CACHE_GB="${mode_disk_cache_gb}" \
  BENCH_PHASE56_MAX_TOKENS="${MAX_TOKENS}" \
  BENCH_PHASE56_TEMPERATURE="${TEMPERATURE}" \
  BENCH_PHASE56_SEED="${SEED}" \
  BENCH_PHASE56_REQUEST_SEED="${REQUEST_SEED}" \
  BENCH_PHASE56_REHYDRATE_POPULATE_SESSIONS="${populate_sessions}" \
  BENCH_PHASE56_REHYDRATE_THRASH_SESSIONS="${thrash_sessions}" \
  BENCH_PHASE56_REHYDRATE_TURNS="${turns}" \
  BENCH_PHASE56_REHYDRATE_PREFIX_TARGET_TOKENS="${prefix_tokens}" \
  BENCH_PHASE56_REHYDRATE_POPULATE_CONC="${PRESSURE_POPULATE_CONC}" \
  BENCH_PHASE56_REHYDRATE_THRASH_CONC="${PRESSURE_THRASH_CONC}" \
  BENCH_PHASE56_REHYDRATE_REPLAY_CONC="${replay_conc}" \
  BENCH_PHASE56_REHYDRATE_REPLAY_REPEATS="${replay_repeats}" \
  BENCH_PHASE56_REHYDRATE_GEN_TOKENS="${gen_tokens}" \
  BENCH_PHASE56_COLLECT_TELEMETRY="0" \
  BENCH_PHASE56_METRICS_SYSTEM_PORT="8081" \
  BENCH_PHASE56_METRICS_KVBM_PORT="6880" \
  BENCH_PHASE56_DISABLE_DISK_OFFLOAD_FILTER="${disable_filter}" \
  BENCH_PHASE56_IO_ATTRIB="${IO_ATTRIB}" \
  BENCH_PHASE56_IO_ATTRIB_INTERVAL_S="${IO_ATTRIB_INTERVAL_S}" \
  BENCH_PHASE56_STREAM_METRICS="${STREAM_METRICS}" \
  BENCH_PHASE56_STREAM_TIMEOUT_S="${STREAM_TIMEOUT_S}" \
  BENCH_PHASE56_STREAM_RECORD_TTFB="${STREAM_RECORD_TTFB}" \
  BENCH_PHASE56_TTFC_SANITY_VALIDATE="${TTFC_SANITY_VALIDATE}" \
  BENCH_PHASE56_TTFC_SANITY_FAIL_ON_ERROR="${TTFC_SANITY_FAIL_ON_ERROR}" \
  BENCH_PHASE56_TTFC_SANITY_REQUESTS="${TTFC_SANITY_REQUESTS}" \
  BENCH_PHASE56_TTFC_SANITY_CONCURRENCY="${TTFC_SANITY_CONCURRENCY}" \
  BENCH_PHASE56_TTFC_SANITY_SHORT_MAX_TOKENS="${TTFC_SANITY_SHORT_MAX_TOKENS}" \
  BENCH_PHASE56_TTFC_SANITY_LONG_MAX_TOKENS="${TTFC_SANITY_LONG_MAX_TOKENS}" \
  BENCH_PHASE56_TTFC_SANITY_TTFC_RATIO_THRESHOLD="${TTFC_SANITY_TTFC_RATIO_THRESHOLD}" \
  BENCH_PHASE56_TTFC_SANITY_TTFT_RATIO_THRESHOLD="${TTFC_SANITY_TTFT_RATIO_THRESHOLD}" \
  DYN_KVBM_DISK_CACHE_DIR="${mode_cache_dir}" \
  BENCH_CONTAINER_NAME="${CONTAINER_NAME}" \
  scripts/bench_phase56_like_probe_trtllm.sh >> "${log_path}" 2>&1
  local rc=$?
  set -e

  if (( rc != 0 )); then
    echo "probe_failed:${mode}:${bundle_id}:rc=${rc}" >&2
    return "${rc}"
  fi

  local run_dir="${RESULTS_ROOT}/${bundle_id}/${run_id}"
  if is_truthy "${IO_ATTRIB}"; then
    {
      echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) io_attrib_checker start mode=${mode} bundle=${bundle_id} ===="
    } >> "${log_path}"
    local io_check_rc=0
    if [[ "${mode}" == "B0" ]]; then
      echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) io_attrib_checker skipped mode=${mode} (strict replay gate disabled for B0) ====" >> "${log_path}"
    else
      set +e
      "${PYTHON_BIN}" "${IO_ATTRIB_CHECKER}" --run-dir "${run_dir}" --expect-report >> "${log_path}" 2>&1
      io_check_rc=$?
      set -e
    fi
    {
      echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) io_attrib_checker done mode=${mode} bundle=${bundle_id} rc=${io_check_rc} ===="
    } >> "${log_path}"
    if (( io_check_rc != 0 )); then
      echo "io_attrib_checker_failed:${mode}:${bundle_id}:rc=${io_check_rc}" >&2
    fi
  fi

  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_probe success mode=${mode} bundle=${bundle_id} ====" >> "${log_path}"
  echo "${run_dir}"
}

mechanism_gate_any_positive() {
  local metrics_json="$1"
  "${PYTHON_BIN}" - "${metrics_json}" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
m = obj.get("mechanism", {})
ok = (
    float(m.get("kvbm_matched_tokens_delta_replay_plus_replay2") or 0.0) > 0.0
    or float(m.get("kvbm_onboard_blocks_d2d_delta_replay_plus_replay2") or 0.0) > 0.0
    or int(m.get("block_read_bytes_delta_replay_plus_replay2") or 0) > 0
    or int(m.get("cgroup_read_bytes_delta_replay_plus_replay2") or 0) > 0
)
print("1" if ok else "0")
PY
}

validate_ttfc_capture() {
  local mode="$1"
  local metrics_json="$2"
  local require_ttfc="${3:-1}"
  "${PYTHON_BIN}" - "${mode}" "${metrics_json}" "${require_ttfc}" <<'PY'
import json
import sys

mode = sys.argv[1]
obj = json.loads(sys.argv[2])
require_ttfc = str(sys.argv[3]).strip().lower() in {"1", "true", "yes", "on"}
if not require_ttfc:
    print("OK")
    raise SystemExit(0)

stream_enabled = bool(obj.get("stream"))
replay_ttfc = obj.get("replay_ttfc_ms") if isinstance(obj.get("replay_ttfc_ms"), dict) else {}
p95 = replay_ttfc.get("p95") if isinstance(replay_ttfc, dict) else None
if not stream_enabled:
    print(f"FAIL:stream_disabled_ttfc_required:mode={mode}")
    raise SystemExit(0)
if p95 is None:
    print(f"FAIL:missing_replay_ttfc_p95:mode={mode}")
    raise SystemExit(0)
print("OK")
PY
}

validate_full_point() {
  local mode="$1"
  local metrics_json="$2"
  "${PYTHON_BIN}" - "${mode}" "${metrics_json}" <<'PY'
import json
import sys

mode = sys.argv[1]
obj = json.loads(sys.argv[2])
errs = []

if obj.get("error_rate") is None or float(obj.get("error_rate")) > 0.0:
    errs.append(f"error_rate={obj.get('error_rate')}")

ttft = obj.get("overall_ttft_ms") or {}
for k in ("p50", "p95", "p99"):
    if ttft.get(k) is None:
        errs.append(f"missing_overall_ttft_{k}")

replay_ttft = obj.get("replay_ttft_ms") or {}
for k in ("p50", "p95", "p99"):
    if replay_ttft.get(k) is None:
        errs.append(f"missing_replay_ttft_{k}")

if mode == "B2":
    m = obj.get("mechanism", {})
    matched = float(m.get("kvbm_matched_tokens_delta_replay_plus_replay2") or 0.0)
    onboard = float(m.get("kvbm_onboard_blocks_d2d_delta_replay_plus_replay2") or 0.0)
    block_read = int(m.get("block_read_bytes_delta_replay_plus_replay2") or 0)
    cgroup_read = int(m.get("cgroup_read_bytes_delta_replay_plus_replay2") or 0)
    if not (matched > 0.0 and onboard > 0.0 and (block_read > 0 or cgroup_read > 0)):
        errs.append(
            f"mechanism_disappeared:matched={matched},onboard={onboard},block_read={block_read},cgroup_read={cgroup_read}"
        )
elif mode == "B1":
    verified = obj.get("b1_disk_tier_verified")
    verification = obj.get("b1_disk_tier_verification") if isinstance(obj.get("b1_disk_tier_verification"), dict) else {}
    enforcement_enabled = bool(verification.get("enforcement_enabled", True))
    fail_reasons = verification.get("fail_reasons") if isinstance(verification.get("fail_reasons"), list) else []
    if enforcement_enabled and verified is not True:
        suffix = ",".join(str(item) for item in fail_reasons if str(item))
        if suffix:
            errs.append(f"B1_DISK_TIER_NOT_DISABLED:{suffix}")
        else:
            errs.append("B1_DISK_TIER_NOT_DISABLED")

for name, d in (obj.get("nvme_artifacts") or {}).items():
    if d.get("success") is not True:
        errs.append(f"{name}_success_false")
    if "payload_ok" not in d:
        errs.append(f"{name}_payload_ok_missing")

if errs:
    print("FAIL:" + ";".join(errs))
else:
    print("OK")
PY
}

validate_io_attrib_point() {
  local mode="$1"
  local metrics_json="$2"
  if ! is_truthy "${IO_ATTRIB}"; then
    echo "OK"
    return 0
  fi
  if [[ "${mode}" == "B0" ]]; then
    echo "OK"
    return 0
  fi
  "${PYTHON_BIN}" - "${metrics_json}" <<'PY'
import json
import sys

obj = json.loads(sys.argv[1])
verdict = obj.get("io_attribution_verdict") if isinstance(obj.get("io_attribution_verdict"), dict) else {}
if not verdict.get("available"):
    reason = verdict.get("error") or "io_attrib_verdict_unavailable"
    print(f"FAIL:{reason}")
    raise SystemExit(0)
if verdict.get("pass") is True:
    print("OK")
    raise SystemExit(0)

failed_checks = list(verdict.get("failed_checks") or [])
if failed_checks:
    print("FAIL:" + ",".join(str(item) for item in failed_checks))
else:
    print("FAIL:io_attrib_replay_gate_failed")
PY
}

emit_diagnosis() {
  local known_run="$1"
  local failed_run="$2"
  local out_json="$3"
  "${PYTHON_BIN}" - "${known_run}" "${failed_run}" "${out_json}" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

known_run = pathlib.Path(sys.argv[1])
failed_run = pathlib.Path(sys.argv[2])
out = pathlib.Path(sys.argv[3])

def load_json(path):
    return json.loads(path.read_text())

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def collect(run_dir: pathlib.Path) -> dict:
    bundle_dir = run_dir.parent
    analysis_dir = bundle_dir / "analysis"
    analysis_manifest = load_json(analysis_dir / "manifest.json")
    runtime_manifest = load_json(analysis_dir / "worker_runtime_manifest.json")
    cfg = load_json(run_dir / "config.json")
    summary = load_json(run_dir / "summary.json")
    phases = {p.get("phase"): p for p in summary.get("phase_summaries", []) if p.get("phase")}
    replay = phases.get("replay", {})
    replay2 = phases.get("replay_2", {})
    io_replay = load_json(run_dir / "phase_deltas/phase_replay_os_io_delta.json")
    io_replay2 = load_json(run_dir / "phase_deltas/phase_replay_2_os_io_delta.json")
    kv_replay = load_json(run_dir / "phase_deltas/phase_replay_kvbm_metrics_delta.json")
    kv_replay2 = load_json(run_dir / "phase_deltas/phase_replay_2_kvbm_metrics_delta.json")
    req = run_dir / "request_manifest.jsonl"
    prm = run_dir / "prompts_manifest.jsonl"
    return {
        "run_path": str(run_dir),
        "bundle_path": str(bundle_dir),
        "analysis_manifest": {
            "scenario": analysis_manifest.get("scenario"),
            "tier_mode": analysis_manifest.get("tier_mode"),
            "kv_mode": analysis_manifest.get("kv_mode"),
            "cpu_cache_gb": ((analysis_manifest.get("kvbm") or {}).get("cpu_cache_gb")),
            "disk_cache_gb": ((analysis_manifest.get("kvbm") or {}).get("disk_cache_gb")),
            "disk_offload_filter_override": ((analysis_manifest.get("kvbm") or {}).get("disk_offload_filter_override")),
        },
        "runtime_env": {
            "DYN_KVBM_DISK_CACHE_DIR": ((runtime_manifest.get("env") or {}).get("DYN_KVBM_DISK_CACHE_DIR")),
            "DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER": ((runtime_manifest.get("env") or {}).get("DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER")),
            "DYN_KVBM_CPU_CACHE_GB": ((runtime_manifest.get("env") or {}).get("DYN_KVBM_CPU_CACHE_GB")),
            "DYN_KVBM_DISK_CACHE_GB": ((runtime_manifest.get("env") or {}).get("DYN_KVBM_DISK_CACHE_GB")),
        },
        "config_args": {
            "scenario": ((cfg.get("args") or {}).get("scenario")),
            "rehydrate_populate_sessions": ((cfg.get("args") or {}).get("rehydrate_populate_sessions")),
            "rehydrate_thrash_sessions": ((cfg.get("args") or {}).get("rehydrate_thrash_sessions")),
            "rehydrate_turns": ((cfg.get("args") or {}).get("rehydrate_turns")),
            "rehydrate_prefix_target_tokens": ((cfg.get("args") or {}).get("rehydrate_prefix_target_tokens")),
            "rehydrate_populate_concurrency": ((cfg.get("args") or {}).get("rehydrate_populate_concurrency")),
            "rehydrate_thrash_concurrency": ((cfg.get("args") or {}).get("rehydrate_thrash_concurrency")),
            "rehydrate_replay_concurrency": ((cfg.get("args") or {}).get("rehydrate_replay_concurrency")),
            "rehydrate_replay_repeats": ((cfg.get("args") or {}).get("rehydrate_replay_repeats")),
            "seed": ((cfg.get("args") or {}).get("seed")),
            "request_seed": ((cfg.get("args") or {}).get("request_seed")),
            "diagnostic_disable_disk_offload_filter": ((cfg.get("args") or {}).get("diagnostic_disable_disk_offload_filter")),
        },
        "manifest_hashes": {
            "request_manifest_sha256": sha256(req),
            "prompts_manifest_sha256": sha256(prm),
            "request_manifest_rows": sum(1 for _ in req.open("r", encoding="utf-8")),
            "prompts_manifest_rows": sum(1 for _ in prm.open("r", encoding="utf-8")),
        },
        "signals": {
            "replay_p95_ms": ((replay.get("latency_ms") or {}).get("p95")),
            "replay_ttft_p95_ms": ((replay.get("ttft_ms") or {}).get("p95")),
            "error_rate": ((summary.get("overall_summary") or {}).get("error_rate")),
            "kvbm_matched_tokens_delta_replay_plus_replay2": (kv_replay.get("kvbm_matched_tokens_delta") or 0.0) + (kv_replay2.get("kvbm_matched_tokens_delta") or 0.0),
            "kvbm_onboard_blocks_d2d_delta_replay_plus_replay2": (kv_replay.get("kvbm_onboard_blocks_d2d_delta") or 0.0) + (kv_replay2.get("kvbm_onboard_blocks_d2d_delta") or 0.0),
            "block_read_bytes_delta_replay_plus_replay2": ((io_replay.get("block_device_delta") or {}).get("read_bytes_delta") or 0) + ((io_replay2.get("block_device_delta") or {}).get("read_bytes_delta") or 0),
            "cgroup_read_bytes_delta_replay_plus_replay2": ((io_replay.get("worker_process_io_delta") or {}).get("cgroup_read_bytes_delta") or 0) + ((io_replay2.get("worker_process_io_delta") or {}).get("cgroup_read_bytes_delta") or 0),
        },
    }

known = collect(known_run)
failed = collect(failed_run)

def diff_dict(a: dict, b: dict):
    out = {}
    for k in sorted(set(a) | set(b)):
        av = a.get(k)
        bv = b.get(k)
        if av != bv:
            out[k] = {"known_good": av, "failed_b2_c1": bv}
    return out

payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "known_good_run": known["run_path"],
    "failed_run": failed["run_path"],
    "known_good": known,
    "failed_b2_c1": failed,
    "diff": {
        "analysis_manifest": diff_dict(known["analysis_manifest"], failed["analysis_manifest"]),
        "runtime_env": diff_dict(known["runtime_env"], failed["runtime_env"]),
        "config_args": diff_dict(known["config_args"], failed["config_args"]),
        "manifest_hashes": diff_dict(known["manifest_hashes"], failed["manifest_hashes"]),
        "signals": diff_dict(known["signals"], failed["signals"]),
    },
    "finding_summary": [],
}

if payload["diff"]["manifest_hashes"]:
    payload["finding_summary"].append("request/prompts manifests diverged from known-good")
else:
    payload["finding_summary"].append("request/prompts manifests match known-good")

if payload["diff"]["config_args"]:
    payload["finding_summary"].append("config args differ from known-good (notably concurrency fields)")

if failed["signals"]["kvbm_matched_tokens_delta_replay_plus_replay2"] == 0 and failed["signals"]["kvbm_onboard_blocks_d2d_delta_replay_plus_replay2"] == 0:
    payload["finding_summary"].append("failed B2@c1 shows no reuse/onboard mechanism signal")

out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(out)
PY
}

run_debug_b2_c1_if_needed() {
  local reason="$1"
  local cache_dir="${KVBM_CACHE_BASE_DIR}/debug_b2_c1"
  local bundle_id="phase60_rehydrate_minimal_debug_B2_c1_${TS}"
  local run_id="run_B2_c1_debug_${TS}"
  local run_dir
  run_dir="$(run_probe "B2" "${bundle_id}" "${run_id}" "1" "${cache_dir}" "true" "${POP_SESSIONS}" "${THRASH_SESSIONS}" "${TURNS}" "${PREFIX_TOKENS}" "${REPLAY_REPEATS}" "${REHYDRATE_GEN_TOKENS}")" || return 0
  local metrics_json
  metrics_json="$(collect_run_metrics "${run_dir}")"
  local row_json
  row_json="$("${PYTHON_BIN}" - "${metrics_json}" "${reason}" <<'PY'
import json
import sys
metrics = json.loads(sys.argv[1])
reason = sys.argv[2]
row = {
    "mode": "B2",
    "concurrency": 1,
    "phase": "debug_filter_disabled",
    "status": "debug_only",
    "debug_reason": reason,
    "disk_offload_filter_disabled": True,
    **metrics,
}
print(json.dumps(row, separators=(",", ":")))
PY
)"
  append_row "${row_json}"
}

finalize_sweep_success() {
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" <<'PY'
import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
obj = json.loads(p.read_text())
rows = [r for r in obj.get("rows", []) if r.get("status") == "ok"]

baseline = obj.get("baseline_b2_replay_p95_ms_at_concurrency1")
if baseline is None:
    for r in rows:
        if r.get("mode") == "B2" and int(r.get("concurrency", -1)) == 1:
            baseline = ((r.get("replay_latency_ms") or {}).get("p95"))
            break

if baseline is None:
    obj["status"] = "stopped"
    obj["decision_grade"] = False
    obj["stop_reason"] = "baseline_missing"
    obj["stop_detail"] = "Missing valid B2@c1 row"
    obj["baseline_b2_replay_p95_ms_at_concurrency1"] = None
    obj["slo_replay_p95_ms"] = None
    obj["max_concurrency_meeting_slo"] = {"B0": None, "B1": None, "B2": None}
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0)

slo = float(baseline) * 1.10
obj["baseline_b2_replay_p95_ms_at_concurrency1"] = baseline
obj["slo_replay_p95_ms"] = slo

mx = {"B0": 0, "B1": 0, "B2": 0}
for r in obj.get("rows", []):
    if r.get("status") != "ok":
        r["pass_slo"] = None
        continue
    rp95 = ((r.get("replay_latency_ms") or {}).get("p95"))
    er = r.get("error_rate")
    passed = bool(rp95 is not None and er is not None and float(er) <= 0.0 and float(rp95) <= slo)
    r["pass_slo"] = passed
    mode = r.get("mode")
    if passed and mode in mx:
        mx[mode] = max(mx[mode], int(r.get("concurrency") or 0))

obj["max_concurrency_meeting_slo"] = mx
metric_counts = {"replay_ttfc_p95_ms": 0, "replay_ttft_p95_ms": 0, "missing": 0}
for r in rows:
    metric_used = r.get("metric_used") if isinstance(r.get("metric_used"), dict) else {}
    replay_p95_used = metric_used.get("replay_p95")
    if replay_p95_used in metric_counts:
        metric_counts[replay_p95_used] += 1
        continue
    if replay_p95_used == "ttfc_ms":
        metric_counts["replay_ttfc_p95_ms"] += 1
        continue
    if replay_p95_used == "ttft_ms":
        metric_counts["replay_ttft_p95_ms"] += 1
        continue

    replay_ttfc = ((r.get("replay_ttfc_ms") or {}).get("p95"))
    replay_ttft = ((r.get("replay_ttft_ms") or {}).get("p95"))
    if replay_ttfc is not None:
        metric_counts["replay_ttfc_p95_ms"] += 1
    elif replay_ttft is not None:
        metric_counts["replay_ttft_p95_ms"] += 1
    else:
        metric_counts["missing"] += 1

dominant_metric = "mixed"
non_zero = [name for name, count in metric_counts.items() if count > 0]
if len(non_zero) == 1:
    dominant_metric = non_zero[0]
elif not non_zero:
    dominant_metric = "missing"
meta = obj.setdefault("meta", {})
meta["metric_policy"] = {
    "preferred": "replay_ttfc_p95_ms",
    "fallback": "replay_ttft_p95_ms",
    "used_replay_p95": dominant_metric,
    "used_replay_p95_counts": metric_counts,
}
warnings = obj.get("warnings")
warning_count = len(warnings) if isinstance(warnings, list) else 0
obj["warning_count"] = warning_count
if warning_count > 0:
    obj["status"] = "completed"
    obj["decision_grade"] = False
else:
    obj["status"] = "completed"
    obj["decision_grade"] = True
p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

emit_summary_csv() {
  if [[ ! -f "${SWEEP_SUMMARY_JSON}" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${SWEEP_SUMMARY_CSV}" <<'PY'
import csv
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])
obj = json.loads(summary_path.read_text())
rows = obj.get("rows") if isinstance(obj.get("rows"), list) else []

fieldnames = [
    "mode",
    "concurrency",
    "status",
    "pass_slo",
    "error_rate",
    "overall_ttfc_p95_ms",
    "overall_ttfc_p99_ms",
    "replay_ttfc_p95_ms",
    "replay_ttfc_p99_ms",
    "overall_ttft_p95_ms",
    "overall_ttft_p99_ms",
    "replay_ttft_p95_ms",
    "replay_ttft_p99_ms",
    "metric_used_replay_p95",
    "populate_read_gib",
    "populate_write_gib",
    "thrash_read_gib",
    "thrash_write_gib",
    "replay_read_gib",
    "replay_write_gib",
    "replay_2_read_gib",
    "replay_2_write_gib",
    "kvbm_metrics_status",
    "kvbm_enabled",
    "kvbm_metrics_available",
    "kvbm_matched_tokens_delta_replay_plus_replay2",
    "kvbm_onboard_blocks_d2d_delta_replay_plus_replay2",
    "b1_disk_tier_verified",
    "run_path",
]

def phase_val(phase_io: dict, phase: str, key: str):
    payload = phase_io.get(phase) if isinstance(phase_io, dict) else {}
    if isinstance(payload, dict):
        value = payload.get(key)
        return "" if value is None else value
    return ""

with csv_path.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode") or "")
        if mode not in {"B0", "B1", "B2"}:
            continue
        kvbm_status = row.get("kvbm_metrics_status") if isinstance(row.get("kvbm_metrics_status"), dict) else {}
        status = str(kvbm_status.get("status") or "")
        kvbm_enabled = bool(row.get("kvbm_enabled"))
        kvbm_available = bool(row.get("kvbm_metrics_available"))
        if not status:
            if not kvbm_enabled:
                status = "skipped"
            elif kvbm_available:
                status = "ok"
            else:
                status = "unavailable"
        kvbm_ok = status == "ok"
        mechanism = row.get("mechanism") if isinstance(row.get("mechanism"), dict) else {}
        phase_io = row.get("phase_io_gib") if isinstance(row.get("phase_io_gib"), dict) else {}
        writer.writerow(
            {
                "mode": mode,
                "concurrency": row.get("concurrency"),
                "status": row.get("status"),
                "pass_slo": row.get("pass_slo"),
                "error_rate": row.get("error_rate"),
                "overall_ttfc_p95_ms": ((row.get("overall_ttfc_ms") or {}).get("p95")),
                "overall_ttfc_p99_ms": ((row.get("overall_ttfc_ms") or {}).get("p99")),
                "replay_ttfc_p95_ms": ((row.get("replay_ttfc_ms") or {}).get("p95")),
                "replay_ttfc_p99_ms": ((row.get("replay_ttfc_ms") or {}).get("p99")),
                "overall_ttft_p95_ms": ((row.get("overall_ttft_ms") or {}).get("p95")),
                "overall_ttft_p99_ms": ((row.get("overall_ttft_ms") or {}).get("p99")),
                "replay_ttft_p95_ms": ((row.get("replay_ttft_ms") or {}).get("p95")),
                "replay_ttft_p99_ms": ((row.get("replay_ttft_ms") or {}).get("p99")),
                "metric_used_replay_p95": ((row.get("metric_used") or {}).get("replay_p95")),
                "populate_read_gib": phase_val(phase_io, "populate", "read_gib"),
                "populate_write_gib": phase_val(phase_io, "populate", "write_gib"),
                "thrash_read_gib": phase_val(phase_io, "thrash", "read_gib"),
                "thrash_write_gib": phase_val(phase_io, "thrash", "write_gib"),
                "replay_read_gib": phase_val(phase_io, "replay", "read_gib"),
                "replay_write_gib": phase_val(phase_io, "replay", "write_gib"),
                "replay_2_read_gib": phase_val(phase_io, "replay_2", "read_gib"),
                "replay_2_write_gib": phase_val(phase_io, "replay_2", "write_gib"),
                "kvbm_metrics_status": status,
                "kvbm_enabled": kvbm_enabled,
                "kvbm_metrics_available": kvbm_available,
                "kvbm_matched_tokens_delta_replay_plus_replay2": (
                    mechanism.get("kvbm_matched_tokens_delta_replay_plus_replay2") if kvbm_ok else ""
                ),
                "kvbm_onboard_blocks_d2d_delta_replay_plus_replay2": (
                    mechanism.get("kvbm_onboard_blocks_d2d_delta_replay_plus_replay2") if kvbm_ok else ""
                ),
                "b1_disk_tier_verified": row.get("b1_disk_tier_verified"),
                "run_path": row.get("run_path"),
            }
        )
print(csv_path)
PY
}

ensure_container
trap 'emit_summary_csv >/dev/null 2>&1 || true' EXIT
emit_diagnosis "${KNOWN_GOOD_B2_RUN}" "${FAILED_B2_C1_RUN}" "${DIAG_JSON}"

if [[ ! -f "${PHASE60_SEMANTIC_HASH_HELPER}" ]]; then
  echo "Missing helper: ${PHASE60_SEMANTIC_HASH_HELPER}" >&2
  exit 2
fi

SEMANTIC_CONTEXT_JSON="$(build_semantic_context_json)"
if [[ -f "${BASELINE_SEMANTIC_HASH_FILE}" ]]; then
  KNOWN_SEMANTIC_PAYLOAD="$(resolve_known_semantic_payload "${BASELINE_SEMANTIC_HASH_FILE}" "{}")"
else
  FALLBACK_KNOWN_SEMANTIC_PAYLOAD="$(compute_semantic_payload_for_run "${KNOWN_GOOD_B2_RUN}" "${SEMANTIC_CONTEXT_JSON}")"
  KNOWN_SEMANTIC_PAYLOAD="$(resolve_known_semantic_payload "${BASELINE_SEMANTIC_HASH_FILE}" "${FALLBACK_KNOWN_SEMANTIC_PAYLOAD}")"
fi
KNOWN_SEMANTIC_HASH="$("${PYTHON_BIN}" - "${KNOWN_SEMANTIC_PAYLOAD}" <<'PY'
import json,sys
payload = json.loads(sys.argv[1])
print(payload.get("semantic_hash") or "")
PY
)"
KNOWN_SEMANTIC_SOURCE="$("${PYTHON_BIN}" - "${KNOWN_SEMANTIC_PAYLOAD}" <<'PY'
import json,sys
payload = json.loads(sys.argv[1])
if payload.get("source") == "baseline_file":
    print(payload.get("baseline_file") or "baseline_file")
else:
    print(payload.get("run_path") or "")
PY
)"

if [[ -f "${SWEEP_SUMMARY_JSON}" ]] && ! is_truthy "${FORCE_NEW_SUMMARY}"; then
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${TS}" "${DIAG_JSON}" "${KNOWN_GOOD_B2_RUN}" "${CONC_LIST}" "${PRESSURE_POPULATE_CONC}" "${PRESSURE_THRASH_CONC}" "${BASELINE_REPLAY_CONC}" "${RESUME_FROM}" "${IO_ATTRIB}" "${IO_ATTRIB_INTERVAL_S}" "${INCLUDE_B0_ENABLED}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sweep_concs = [int(x) for x in str(sys.argv[5]).split() if x.strip()]

def to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

include_b0 = to_bool(sys.argv[12])
run_order = ["B2", "B1"] + (["B0"] if include_b0 else [])

obj = json.loads(path.read_text())
meta = obj.setdefault("meta", {})
meta.update(
    {
        "resumed_utc": sys.argv[2],
        "scenario": "rehydrate_replay",
        "model_profile": "llama31_8b_fp8",
        "seed": 20260210,
        "request_seed": 20260210,
        "sweep_replay_concurrencies": sweep_concs,
        "baseline_replay_concurrency": int(sys.argv[8]),
        "pressure_populate_concurrency": int(sys.argv[6]),
        "pressure_thrash_concurrency": int(sys.argv[7]),
        "io_attrib_enabled": to_bool(sys.argv[10]),
        "io_attrib_interval_s": float(sys.argv[11]),
        "include_b0": include_b0,
        "run_order_per_concurrency": run_order,
        "slo_definition": "replay_p95_ms <= (baseline_B2_replay_p95_ms_at_concurrency1 * 1.10)",
        "metric_policy": {"preferred": "replay_ttfc_p95_ms", "fallback": "replay_ttft_p95_ms", "used_replay_p95": "pending"},
        "diagnosis_json": sys.argv[3],
        "known_good_b2_run": sys.argv[4],
        "resume_from": sys.argv[9] or None,
    }
)
obj["status"] = "running"
obj["decision_grade"] = None
obj["stop_reason"] = None
obj["stop_detail"] = None
obj["stopped_at"] = None
obj.setdefault("rows", [])
obj.setdefault("warnings", [])
obj.setdefault("baseline_b2_replay_p95_ms_at_concurrency1", None)
obj.setdefault("slo_replay_p95_ms", None)
obj.setdefault("max_concurrency_meeting_slo", {"B0": None, "B1": None, "B2": None})
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
else
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${TS}" "${DIAG_JSON}" "${KNOWN_GOOD_B2_RUN}" "${CONC_LIST}" "${PRESSURE_POPULATE_CONC}" "${PRESSURE_THRASH_CONC}" "${BASELINE_REPLAY_CONC}" "${RESUME_FROM}" "${IO_ATTRIB}" "${IO_ATTRIB_INTERVAL_S}" "${INCLUDE_B0_ENABLED}" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
sweep_concs = [int(x) for x in str(sys.argv[5]).split() if x.strip()]

def to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

include_b0 = to_bool(sys.argv[12])
run_order = ["B2", "B1"] + (["B0"] if include_b0 else [])

payload = {
    "meta": {
        "created_utc": sys.argv[2],
        "scenario": "rehydrate_replay",
        "model_profile": "llama31_8b_fp8",
        "seed": 20260210,
        "request_seed": 20260210,
        "sweep_replay_concurrencies": sweep_concs,
        "baseline_replay_concurrency": int(sys.argv[8]),
        "pressure_populate_concurrency": int(sys.argv[6]),
        "pressure_thrash_concurrency": int(sys.argv[7]),
        "io_attrib_enabled": to_bool(sys.argv[10]),
        "io_attrib_interval_s": float(sys.argv[11]),
        "include_b0": include_b0,
        "run_order_per_concurrency": run_order,
        "slo_definition": "replay_p95_ms <= (baseline_B2_replay_p95_ms_at_concurrency1 * 1.10)",
        "metric_policy": {"preferred": "replay_ttfc_p95_ms", "fallback": "replay_ttft_p95_ms", "used_replay_p95": "pending"},
        "diagnosis_json": sys.argv[3],
        "known_good_b2_run": sys.argv[4],
        "resume_from": sys.argv[9] or None,
    },
    "rows": [],
    "warnings": [],
    "status": "running",
    "decision_grade": None,
    "baseline_b2_replay_p95_ms_at_concurrency1": None,
    "slo_replay_p95_ms": None,
    "max_concurrency_meeting_slo": {"B0": None, "B1": None, "B2": None},
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi

summary_set_baseline_policy_meta "${KNOWN_SEMANTIC_HASH}" "${KNOWN_SEMANTIC_SOURCE}"

RESUME_FROM_NORMALIZED=""
resume_gate_open="true"
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ "${RESUME_FROM}" =~ ^([Bb][012])_[Cc]([0-9]+)$ ]]; then
    RESUME_FROM_NORMALIZED="${BASH_REMATCH[1]^^}_c${BASH_REMATCH[2]}"
    resume_gate_open="false"
  else
    echo "Invalid BENCH_PHASE60_RESUME_FROM='${RESUME_FROM}'. Expected like B2_c2 or B0_c2." >&2
    exit 2
  fi
fi

should_run_point() {
  local mode="$1"
  local conc="$2"
  local point_key="${mode}_c${conc}"
  if [[ "${resume_gate_open}" != "true" ]]; then
    if [[ "${point_key}" == "${RESUME_FROM_NORMALIZED}" ]]; then
      resume_gate_open="true"
    else
      return 1
    fi
  fi
  if is_truthy "${RESUME_SKIP_COMPLETED}" && summary_has_status "${point_key}" "ok"; then
    return 1
  fi
  return 0
}

# Single baseline preflight: B2 at replay concurrency=1 with full-pressure shape.
if ! summary_has_baseline; then
  pre_id="phase60_rehydrate_minimal_preflight_B2_c${BASELINE_REPLAY_CONC}_${TS}"
  pre_run="run_B2_c${BASELINE_REPLAY_CONC}_preflight_${TS}"
  pre_cache="${KVBM_CACHE_BASE_DIR}/B2_c${BASELINE_REPLAY_CONC}_preflight"
  pre_run_dir="$(run_probe "B2" "${pre_id}" "${pre_run}" "${BASELINE_REPLAY_CONC}" "${pre_cache}" "false" "${PREFLIGHT_POP_SESSIONS}" "${PREFLIGHT_THRASH_SESSIONS}" "${PREFLIGHT_TURNS}" "${PREFLIGHT_PREFIX_TOKENS}" "${PREFLIGHT_REPLAY_REPEATS}" "${PREFLIGHT_GEN_TOKENS}")" || {
    update_summary_stop "preflight_probe_failed" "B2 baseline preflight command failed" "B2" "${BASELINE_REPLAY_CONC}"
    emit_stop_verdict "preflight_probe_failed" "B2 baseline preflight command failed" "B2" "${BASELINE_REPLAY_CONC}"
    exit 2
  }
  pre_metrics_json="$(collect_run_metrics "${pre_run_dir}")"
  pre_ttfc_valid="$(validate_ttfc_capture "B2" "${pre_metrics_json}" "${REQUIRE_TTFC_ENABLED}")"
  if [[ "${pre_ttfc_valid}" != "OK" ]]; then
    pre_ttfc_row="$("${PYTHON_BIN}" - "${pre_metrics_json}" "${BASELINE_REPLAY_CONC}" "${pre_cache}" "${pre_ttfc_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_ttfc"
m["phase"]="baseline_preflight"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
    summary_upsert_row "${pre_ttfc_row}" "baseline_preflight_B2_c${BASELINE_REPLAY_CONC}"
    update_summary_stop "ttfc_missing_stream_capture" "${pre_ttfc_valid}" "B2" "${BASELINE_REPLAY_CONC}"
    emit_stop_verdict "ttfc_missing_stream_capture" "${pre_ttfc_valid}" "B2" "${BASELINE_REPLAY_CONC}"
    exit 2
  fi
pre_io_valid="$(validate_io_attrib_point "B2" "${pre_metrics_json}")"
  if [[ "${pre_io_valid}" != "OK" ]]; then
    pre_io_row="$("${PYTHON_BIN}" - "${pre_metrics_json}" "${BASELINE_REPLAY_CONC}" "${pre_cache}" "${pre_io_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_io_attrib"
m["phase"]="baseline_preflight"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
    summary_upsert_row "${pre_io_row}" "baseline_preflight_B2_c${BASELINE_REPLAY_CONC}"
    update_summary_stop "io_attrib_replay_gate_failed" "${pre_io_valid}" "B2" "${BASELINE_REPLAY_CONC}"
    emit_stop_verdict "io_attrib_replay_gate_failed" "${pre_io_valid}" "B2" "${BASELINE_REPLAY_CONC}"
    exit 2
  fi
  pre_ok="$(mechanism_gate_any_positive "${pre_metrics_json}")"
  if [[ "${pre_ok}" != "1" ]]; then
    pre_row="$("${PYTHON_BIN}" - "${pre_metrics_json}" "${BASELINE_REPLAY_CONC}" "${pre_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_preflight"
m["phase"]="baseline_preflight"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
    summary_upsert_row "${pre_row}" "baseline_preflight_B2_c${BASELINE_REPLAY_CONC}"
    update_summary_stop "preflight_mechanism_missing" "B2 baseline preflight had no matched/onboard/read signal" "B2" "${BASELINE_REPLAY_CONC}"
    emit_stop_verdict "preflight_mechanism_missing" "B2 baseline preflight had no matched/onboard/read signal" "B2" "${BASELINE_REPLAY_CONC}"
    if [[ "${BASELINE_REPLAY_CONC}" == "1" ]]; then
      run_debug_b2_c1_if_needed "preflight_mechanism_missing"
    fi
    exit 2
  fi

  baseline_row="$("${PYTHON_BIN}" - "${pre_metrics_json}" "${BASELINE_REPLAY_CONC}" "${pre_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="baseline_preflight"
m["phase"]="baseline_preflight"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
  summary_upsert_row "${baseline_row}" "baseline_preflight_B2_c${BASELINE_REPLAY_CONC}"
  baseline_set="$(summary_set_baseline_from_metrics "${pre_metrics_json}")"
  if [[ "${baseline_set}" != "1" ]]; then
    update_summary_stop "baseline_missing_replay_p95" "B2 baseline preflight missing replay p95 for SLO computation" "B2" "${BASELINE_REPLAY_CONC}"
    emit_stop_verdict "baseline_missing_replay_p95" "B2 baseline preflight missing replay p95 for SLO computation" "B2" "${BASELINE_REPLAY_CONC}"
    exit 2
  fi
fi

for conc in ${CONC_LIST}; do
  point_key_b2="B2_c${conc}"
  if should_run_point "B2" "${conc}"; then
    b2_bundle="phase60_rehydrate_minimal_sweep_B2_c${conc}_${TS}"
    b2_run="run_B2_c${conc}_${TS}"
    b2_cache="${KVBM_CACHE_BASE_DIR}/B2_c${conc}_full"
    b2_run_dir="$(run_probe "B2" "${b2_bundle}" "${b2_run}" "${conc}" "${b2_cache}" "false" "${POP_SESSIONS}" "${THRASH_SESSIONS}" "${TURNS}" "${PREFIX_TOKENS}" "${REPLAY_REPEATS}" "${REHYDRATE_GEN_TOKENS}")" || {
      update_summary_stop "probe_failed" "B2 full point failed to execute" "B2" "${conc}"
      emit_stop_verdict "probe_failed" "B2 full point failed to execute" "B2" "${conc}"
      exit 2
    }
    b2_metrics_json="$(collect_run_metrics "${b2_run_dir}")"
    b2_partial_row="$("${PYTHON_BIN}" - "${b2_metrics_json}" "${conc}" "${b2_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="partial"
m["phase"]="sweep_point"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
    summary_upsert_row "${b2_partial_row}" "${point_key_b2}"

    b2_ttfc_valid="$(validate_ttfc_capture "B2" "${b2_metrics_json}" "${REQUIRE_TTFC_ENABLED}")"
    if [[ "${b2_ttfc_valid}" != "OK" ]]; then
      b2_invalid_ttfc_row="$("${PYTHON_BIN}" - "${b2_metrics_json}" "${conc}" "${b2_cache}" "${b2_ttfc_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_ttfc"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b2_invalid_ttfc_row}" "${point_key_b2}"
      update_summary_stop "ttfc_missing_stream_capture" "${b2_ttfc_valid}" "B2" "${conc}"
      emit_stop_verdict "ttfc_missing_stream_capture" "${b2_ttfc_valid}" "B2" "${conc}"
      exit 2
    fi

    b2_io_valid="$(validate_io_attrib_point "B2" "${b2_metrics_json}")"
    if [[ "${b2_io_valid}" != "OK" ]]; then
      b2_invalid_io_row="$("${PYTHON_BIN}" - "${b2_metrics_json}" "${conc}" "${b2_cache}" "${b2_io_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_io_attrib"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b2_invalid_io_row}" "${point_key_b2}"
      update_summary_stop "io_attrib_replay_gate_failed" "${b2_io_valid}" "B2" "${conc}"
      emit_stop_verdict "io_attrib_replay_gate_failed" "${b2_io_valid}" "B2" "${conc}"
      exit 2
    fi

    b2_valid="$(validate_full_point "B2" "${b2_metrics_json}")"
    if [[ "${b2_valid}" != "OK" ]]; then
      b2_invalid_row="$("${PYTHON_BIN}" - "${b2_metrics_json}" "${conc}" "${b2_cache}" "${b2_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_full"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b2_invalid_row}" "${point_key_b2}"
      update_summary_stop "mechanism_disappeared" "${b2_valid}" "B2" "${conc}"
      emit_stop_verdict "mechanism_disappeared" "${b2_valid}" "B2" "${conc}"
      if [[ "${conc}" == "1" ]]; then
        run_debug_b2_c1_if_needed "${b2_valid}"
      fi
      exit 2
    fi

    if [[ "${conc}" == "1" ]]; then
      b2_semantic_payload="$(compute_semantic_payload_for_run "${b2_run_dir}" "${SEMANTIC_CONTEXT_JSON}")"
      semantic_decision_json="$("${PYTHON_BIN}" "${PHASE60_SEMANTIC_HASH_HELPER}" decide --known-json "${KNOWN_SEMANTIC_PAYLOAD}" --current-json "${b2_semantic_payload}" --strict "${STRICT_BASELINE_HASH_ENABLED}" --accept-new "${ACCEPT_NEW_BASELINE_MANIFEST_ENABLED}")"
      semantic_action="$("${PYTHON_BIN}" - "${semantic_decision_json}" <<'PY'
import json,sys
print((json.loads(sys.argv[1]).get("action") or ""))
PY
)"
      if [[ "${semantic_action}" != "match" ]]; then
        mismatch_diagnostics_json="$("${PYTHON_BIN}" - "${semantic_decision_json}" "${KNOWN_SEMANTIC_PAYLOAD}" "${b2_semantic_payload}" "${BASELINE_SEMANTIC_HASH_FILE}" "${KNOWN_SEMANTIC_SOURCE}" <<'PY'
import json,sys
decision = json.loads(sys.argv[1])
known = json.loads(sys.argv[2])
current = json.loads(sys.argv[3])
diag = {
    "baseline_manifest_path": current.get("manifest_path"),
    "known_good_manifest_path": known.get("manifest_path"),
    "known_good_source": sys.argv[5] or known.get("source"),
    "semantic_hash_file": sys.argv[4],
    "computed_semantic_hash": decision.get("current_hash"),
    "known_good_semantic_hash": decision.get("known_hash"),
    "differing_fields": decision.get("diff_fields") or [],
    "known_run_path": known.get("run_path"),
    "current_run_path": current.get("run_path"),
}
print(json.dumps(diag, separators=(",", ":"), ensure_ascii=True))
PY
)"

        if [[ "${semantic_action}" == "accept" ]]; then
          "${PYTHON_BIN}" "${PHASE60_SEMANTIC_HASH_HELPER}" accept \
            --baseline-file "${BASELINE_SEMANTIC_HASH_FILE}" \
            --audit-jsonl "${BASELINE_SEMANTIC_AUDIT_JSONL}" \
            --known-json "${KNOWN_SEMANTIC_PAYLOAD}" \
            --current-json "${b2_semantic_payload}" \
            --reason "BENCH_PHASE60_ACCEPT_NEW_BASELINE_MANIFEST=1" >/dev/null
          KNOWN_SEMANTIC_PAYLOAD="${b2_semantic_payload}"
          KNOWN_SEMANTIC_HASH="$("${PYTHON_BIN}" - "${KNOWN_SEMANTIC_PAYLOAD}" <<'PY'
import json,sys
print((json.loads(sys.argv[1]).get("semantic_hash") or ""))
PY
)"
          KNOWN_SEMANTIC_SOURCE="${BASELINE_SEMANTIC_HASH_FILE}"
          summary_set_baseline_policy_meta "${KNOWN_SEMANTIC_HASH}" "${KNOWN_SEMANTIC_SOURCE}"
          summary_append_warning "BASELINE_MANIFEST_HASH_ACCEPTED" "B2@c1 semantic baseline hash mismatch accepted and baseline updated" "B2" "1" "${mismatch_diagnostics_json}"
        elif [[ "${semantic_action}" == "warn" ]]; then
          summary_append_warning "BASELINE_MANIFEST_HASH_MISMATCH_WARNING" "B2@c1 semantic baseline hash mismatch (strict mode disabled); continuing sweep" "B2" "1" "${mismatch_diagnostics_json}"
        else
          b2_invalid_hash_row="$("${PYTHON_BIN}" - "${b2_metrics_json}" "${conc}" "${b2_cache}" "${KNOWN_SEMANTIC_HASH}" "${mismatch_diagnostics_json}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_manifest_hash"
m["phase"]="sweep_point"
m["expected_known_good_semantic_hash"]=sys.argv[4]
m["baseline_manifest_hash_diagnostics"]=json.loads(sys.argv[5])
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
          summary_upsert_row "${b2_invalid_hash_row}" "${point_key_b2}"
          update_summary_stop "baseline_manifest_hash_mismatch" "B2@c1 semantic baseline hash mismatch vs known-good" "B2" "1" "${mismatch_diagnostics_json}"
          emit_stop_verdict "baseline_manifest_hash_mismatch" "B2@c1 semantic baseline hash mismatch vs known-good" "B2" "1" "${mismatch_diagnostics_json}"
          exit 2
        fi
      fi
    fi

    b2_ok_row="$("${PYTHON_BIN}" - "${b2_metrics_json}" "${conc}" "${b2_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="ok"
m["phase"]="sweep_point"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
    summary_upsert_row "${b2_ok_row}" "${point_key_b2}"
  fi

  point_key_b1="B1_c${conc}"
  if should_run_point "B1" "${conc}"; then
    b1_bundle="phase60_rehydrate_minimal_sweep_B1_c${conc}_${TS}"
    b1_run="run_B1_c${conc}_${TS}"
    b1_cache="${KVBM_CACHE_BASE_DIR}/B1_c${conc}_full"
    if [[ "${ENFORCE_B1_DISK_TIER_OFF_ENABLED}" == "1" ]]; then
      b1_cache="${B1_KVBM_CACHE_DIR}/c${conc}"
    fi
    b1_run_dir="$(run_probe "B1" "${b1_bundle}" "${b1_run}" "${conc}" "${b1_cache}" "false" "${POP_SESSIONS}" "${THRASH_SESSIONS}" "${TURNS}" "${PREFIX_TOKENS}" "${REPLAY_REPEATS}" "${REHYDRATE_GEN_TOKENS}")" || {
      update_summary_stop "probe_failed" "B1 full point failed to execute" "B1" "${conc}"
      emit_stop_verdict "probe_failed" "B1 full point failed to execute" "B1" "${conc}"
      exit 2
    }
    b1_metrics_json="$(collect_run_metrics "${b1_run_dir}")"
    b1_partial_row="$("${PYTHON_BIN}" - "${b1_metrics_json}" "${conc}" "${b1_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B1"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="partial"
m["phase"]="sweep_point"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
    summary_upsert_row "${b1_partial_row}" "${point_key_b1}"

    b1_ttfc_valid="$(validate_ttfc_capture "B1" "${b1_metrics_json}" "${REQUIRE_TTFC_ENABLED}")"
    if [[ "${b1_ttfc_valid}" != "OK" ]]; then
      b1_invalid_ttfc_row="$("${PYTHON_BIN}" - "${b1_metrics_json}" "${conc}" "${b1_cache}" "${b1_ttfc_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B1"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_ttfc"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b1_invalid_ttfc_row}" "${point_key_b1}"
      update_summary_stop "ttfc_missing_stream_capture" "${b1_ttfc_valid}" "B1" "${conc}"
      emit_stop_verdict "ttfc_missing_stream_capture" "${b1_ttfc_valid}" "B1" "${conc}"
      exit 2
    fi

    b1_io_valid="$(validate_io_attrib_point "B1" "${b1_metrics_json}")"
    if [[ "${b1_io_valid}" != "OK" ]]; then
      b1_invalid_io_row="$("${PYTHON_BIN}" - "${b1_metrics_json}" "${conc}" "${b1_cache}" "${b1_io_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B1"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_io_attrib"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b1_invalid_io_row}" "${point_key_b1}"
      update_summary_stop "io_attrib_replay_gate_failed" "${b1_io_valid}" "B1" "${conc}"
      emit_stop_verdict "io_attrib_replay_gate_failed" "${b1_io_valid}" "B1" "${conc}"
      exit 2
    fi

    b1_valid="$(validate_full_point "B1" "${b1_metrics_json}")"
    if [[ "${b1_valid}" != "OK" ]]; then
      b1_invalid_row="$("${PYTHON_BIN}" - "${b1_metrics_json}" "${conc}" "${b1_cache}" "${b1_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B1"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_full"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b1_invalid_row}" "${point_key_b1}"
      update_summary_stop "point_invalid" "${b1_valid}" "B1" "${conc}"
      emit_stop_verdict "point_invalid" "${b1_valid}" "B1" "${conc}"
      exit 2
    fi

    b1_ok_row="$("${PYTHON_BIN}" - "${b1_metrics_json}" "${conc}" "${b1_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B1"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="ok"
m["phase"]="sweep_point"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
    summary_upsert_row "${b1_ok_row}" "${point_key_b1}"
  fi

  if [[ "${INCLUDE_B0_ENABLED}" == "1" ]]; then
    point_key_b0="B0_c${conc}"
    if should_run_point "B0" "${conc}"; then
      b0_bundle="phase60_rehydrate_minimal_sweep_B0_c${conc}_${TS}"
      b0_run="run_B0_c${conc}_${TS}"
      b0_cache=""
      b0_run_dir="$(run_probe "B0" "${b0_bundle}" "${b0_run}" "${conc}" "${b0_cache}" "false" "${POP_SESSIONS}" "${THRASH_SESSIONS}" "${TURNS}" "${PREFIX_TOKENS}" "${REPLAY_REPEATS}" "${REHYDRATE_GEN_TOKENS}")" || {
        update_summary_stop "probe_failed" "B0 full point failed to execute" "B0" "${conc}"
        emit_stop_verdict "probe_failed" "B0 full point failed to execute" "B0" "${conc}"
        exit 2
      }
      b0_metrics_json="$(collect_run_metrics "${b0_run_dir}")"
      b0_partial_row="$("${PYTHON_BIN}" - "${b0_metrics_json}" "${conc}" "${b0_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B0"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="partial"
m["phase"]="sweep_point"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b0_partial_row}" "${point_key_b0}"

      b0_ttfc_valid="$(validate_ttfc_capture "B0" "${b0_metrics_json}" "${REQUIRE_TTFC_ENABLED}")"
      if [[ "${b0_ttfc_valid}" != "OK" ]]; then
        b0_invalid_ttfc_row="$("${PYTHON_BIN}" - "${b0_metrics_json}" "${conc}" "${b0_cache}" "${b0_ttfc_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B0"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_ttfc"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
        summary_upsert_row "${b0_invalid_ttfc_row}" "${point_key_b0}"
        update_summary_stop "ttfc_missing_stream_capture" "${b0_ttfc_valid}" "B0" "${conc}"
        emit_stop_verdict "ttfc_missing_stream_capture" "${b0_ttfc_valid}" "B0" "${conc}"
        exit 2
      fi

      b0_io_valid="$(validate_io_attrib_point "B0" "${b0_metrics_json}")"
      if [[ "${b0_io_valid}" != "OK" ]]; then
        b0_invalid_io_row="$("${PYTHON_BIN}" - "${b0_metrics_json}" "${conc}" "${b0_cache}" "${b0_io_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B0"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_io_attrib"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
        summary_upsert_row "${b0_invalid_io_row}" "${point_key_b0}"
        update_summary_stop "io_attrib_replay_gate_failed" "${b0_io_valid}" "B0" "${conc}"
        emit_stop_verdict "io_attrib_replay_gate_failed" "${b0_io_valid}" "B0" "${conc}"
        exit 2
      fi

      b0_valid="$(validate_full_point "B0" "${b0_metrics_json}")"
      if [[ "${b0_valid}" != "OK" ]]; then
        b0_invalid_row="$("${PYTHON_BIN}" - "${b0_metrics_json}" "${conc}" "${b0_cache}" "${b0_valid}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B0"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_full"
m["phase"]="sweep_point"
m["validation"]=sys.argv[4]
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
        summary_upsert_row "${b0_invalid_row}" "${point_key_b0}"
        update_summary_stop "point_invalid" "${b0_valid}" "B0" "${conc}"
        emit_stop_verdict "point_invalid" "${b0_valid}" "B0" "${conc}"
        exit 2
      fi

      b0_ok_row="$("${PYTHON_BIN}" - "${b0_metrics_json}" "${conc}" "${b0_cache}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B0"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="ok"
m["phase"]="sweep_point"
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
      summary_upsert_row "${b0_ok_row}" "${point_key_b0}"
    fi
  fi
done

finalize_sweep_success
emit_summary_csv >/dev/null

echo "DIAGNOSIS_JSON=${DIAG_JSON}"
echo "SWEEP_SUMMARY_JSON=${SWEEP_SUMMARY_JSON}"
echo "SWEEP_SUMMARY_CSV=${SWEEP_SUMMARY_CSV}"
echo "STOP_VERDICT_JSON=${STOP_VERDICT_JSON}"
echo "LOG_DIR=${LOG_DIR}"
