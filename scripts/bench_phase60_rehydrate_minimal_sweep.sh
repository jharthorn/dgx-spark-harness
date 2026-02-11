#!/usr/bin/env bash
set -euo pipefail

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

# Preflight defaults inherit full-shape pressure so mechanism signals are comparable.
PREFLIGHT_POP_SESSIONS="${BENCH_PHASE60_PREFLIGHT_POPULATE_SESSIONS:-${POP_SESSIONS}}"
PREFLIGHT_THRASH_SESSIONS="${BENCH_PHASE60_PREFLIGHT_THRASH_SESSIONS:-${THRASH_SESSIONS}}"
PREFLIGHT_TURNS="${BENCH_PHASE60_PREFLIGHT_TURNS:-${TURNS}}"
PREFLIGHT_PREFIX_TOKENS="${BENCH_PHASE60_PREFLIGHT_PREFIX_TARGET_TOKENS:-${PREFIX_TOKENS}}"
PREFLIGHT_REPLAY_REPEATS="${BENCH_PHASE60_PREFLIGHT_REPLAY_REPEATS:-${REPLAY_REPEATS}}"
PREFLIGHT_GEN_TOKENS="${BENCH_PHASE60_PREFLIGHT_GEN_TOKENS:-${REHYDRATE_GEN_TOKENS}}"

KNOWN_GOOD_B2_RUN="${BENCH_PHASE60_KNOWN_GOOD_B2_RUN:-${RESULTS_ROOT}/phase60_rehydrate_B2_r05_20260210T230915Z/run_B2_r05_20260210T230915Z}"
FAILED_B2_C1_RUN="${BENCH_PHASE60_FAILED_B2_C1_RUN:-${RESULTS_ROOT}/phase60_rehydrate_sweep_B2_c1_20260211T000318Z/run_B2_c1_20260211T000318Z}"

DIAG_JSON="${RESULTS_ROOT}/phase60_sweep_b2c1_failure_diagnosis_${TS}.json"
SWEEP_SUMMARY_JSON="${RESULTS_ROOT}/phase60_rehydrate_concurrency_sweep_summary_minimal_${TS}.json"
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
  "${PYTHON_BIN}" - "${STOP_VERDICT_JSON}" "${reason}" "${detail}" "${mode}" "${conc}" "${SWEEP_SUMMARY_JSON}" <<'PY'
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
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${reason}" "${detail}" "${mode}" "${conc}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
reason = sys.argv[2]
detail = sys.argv[3]
mode = sys.argv[4] or None
conc = int(sys.argv[5]) if sys.argv[5] else None
obj = json.loads(path.read_text())
obj["status"] = "stopped"
obj["decision_grade"] = False
obj["stop_reason"] = reason
obj["stop_detail"] = detail
obj["stopped_at"] = {"mode": mode, "concurrency": conc}
obj.setdefault("baseline_b2_replay_p95_ms_at_concurrency1", None)
obj.setdefault("slo_replay_p95_ms", None)
obj.setdefault("max_concurrency_meeting_slo", {"B1": None, "B2": None})
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
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
  "${PYTHON_BIN}" - "${run_dir}" <<'PY'
import hashlib
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
summary = json.loads((run_dir / "summary.json").read_text())
config = json.loads((run_dir / "config.json").read_text())
manifest = json.loads((run_dir / "manifest.json").read_text())

phases = {p.get("phase"): p for p in summary.get("phase_summaries", []) if p.get("phase")}
overall = summary.get("overall_summary", {})
ttft = overall.get("ttft_ms") or {}
replay_ttft = (phases.get("replay", {}) or {}).get("ttft_ms") or {}
replay_lat = (phases.get("replay", {}) or {}).get("latency_ms") or {}

def load_json(path):
    return json.loads(path.read_text())

replay_kv = load_json(run_dir / "phase_deltas/phase_replay_kvbm_metrics_delta.json")
replay2_kv_path = run_dir / "phase_deltas/phase_replay_2_kvbm_metrics_delta.json"
replay2_kv = load_json(replay2_kv_path) if replay2_kv_path.exists() else {}
replay_io = load_json(run_dir / "phase_deltas/phase_replay_os_io_delta.json")
replay2_io_path = run_dir / "phase_deltas/phase_replay_2_os_io_delta.json"
replay2_io = load_json(replay2_io_path) if replay2_io_path.exists() else {}

rb = replay_io.get("block_device_delta", {})
rb2 = replay2_io.get("block_device_delta", {})
wb = replay_io.get("worker_process_io_delta", {})
wb2 = replay2_io.get("worker_process_io_delta", {})

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

payload = {
    "run_path": str(run_dir),
    "error_rate": overall.get("error_rate"),
    "overall_ttft_ms": {
        "p50": ttft.get("p50"),
        "p95": ttft.get("p95"),
        "p99": ttft.get("p99"),
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
    "mechanism": {
        "kvbm_matched_tokens_delta_replay_plus_replay2": (replay_kv.get("kvbm_matched_tokens_delta") or 0.0) + (replay2_kv.get("kvbm_matched_tokens_delta") or 0.0),
        "kvbm_onboard_blocks_d2d_delta_replay_plus_replay2": (replay_kv.get("kvbm_onboard_blocks_d2d_delta") or 0.0) + (replay2_kv.get("kvbm_onboard_blocks_d2d_delta") or 0.0),
        "block_read_bytes_delta_replay_plus_replay2": (rb.get("read_bytes_delta") or 0) + (rb2.get("read_bytes_delta") or 0),
        "block_write_bytes_delta_replay_plus_replay2": (rb.get("write_bytes_delta") or 0) + (rb2.get("write_bytes_delta") or 0),
        "cgroup_read_bytes_delta_replay_plus_replay2": (wb.get("cgroup_read_bytes_delta") or 0) + (wb2.get("cgroup_read_bytes_delta") or 0),
        "cgroup_write_bytes_delta_replay_plus_replay2": (wb.get("cgroup_write_bytes_delta") or 0) + (wb2.get("cgroup_write_bytes_delta") or 0),
    },
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
        "diagnostic_disable_disk_offload_filter": ((config.get("args") or {}).get("diagnostic_disable_disk_offload_filter")),
    },
    "nvme_artifacts": {},
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

print(json.dumps(payload, separators=(",", ":")))
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

  mk_cache_dir "${cache_dir}"
  {
    echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_probe start mode=${mode} bundle=${bundle_id} run=${run_id} replay_conc=${replay_conc} pressure_populate_conc=${PRESSURE_POPULATE_CONC} pressure_thrash_conc=${PRESSURE_THRASH_CONC} cache_dir=${cache_dir} disable_filter=${disable_filter} ===="
  } >> "${log_path}"

  set +e
  BENCH_RESULTS_ROOT="${RESULTS_ROOT}" \
  BENCH_PHASE56_LIKE_BUNDLE_ID="${bundle_id}" \
  BENCH_PHASE56_LIKE_RUN_ID="${run_id}" \
  BENCH_TIER_MODE="${mode}" \
  BENCH_MODEL_PROFILE="${MODEL_PROFILE}" \
  BENCH_PHASE56_SCENARIO="${SCENARIO}" \
  BENCH_PHASE56_CPU_CACHE_GB="${CPU_CACHE_GB}" \
  BENCH_PHASE56_DISK_CACHE_GB="${DISK_CACHE_GB}" \
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
  DYN_KVBM_DISK_CACHE_DIR="${cache_dir}" \
  BENCH_CONTAINER_NAME="${CONTAINER_NAME}" \
  scripts/bench_phase56_like_probe_trtllm.sh >> "${log_path}" 2>&1
  local rc=$?
  set -e

  if (( rc != 0 )); then
    echo "probe_failed:${mode}:${bundle_id}:rc=${rc}" >&2
    return "${rc}"
  fi

  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_probe success mode=${mode} bundle=${bundle_id} ====" >> "${log_path}"
  echo "${RESULTS_ROOT}/${bundle_id}/${run_id}"
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
    obj["max_concurrency_meeting_slo"] = {"B1": None, "B2": None}
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0)

slo = float(baseline) * 1.10
obj["baseline_b2_replay_p95_ms_at_concurrency1"] = baseline
obj["slo_replay_p95_ms"] = slo

mx = {"B1": 0, "B2": 0}
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
obj["status"] = "completed"
obj["decision_grade"] = True
p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
}

ensure_container
emit_diagnosis "${KNOWN_GOOD_B2_RUN}" "${FAILED_B2_C1_RUN}" "${DIAG_JSON}"

KNOWN_METRICS_JSON="$(collect_run_metrics "${KNOWN_GOOD_B2_RUN}")"
KNOWN_REQ_HASH="$("${PYTHON_BIN}" - "${KNOWN_METRICS_JSON}" <<'PY'
import json,sys
print((json.loads(sys.argv[1]).get("manifest_hashes") or {}).get("request_manifest_sha256",""))
PY
)"
KNOWN_PROMPTS_HASH="$("${PYTHON_BIN}" - "${KNOWN_METRICS_JSON}" <<'PY'
import json,sys
print((json.loads(sys.argv[1]).get("manifest_hashes") or {}).get("prompts_manifest_sha256",""))
PY
)"

if [[ -f "${SWEEP_SUMMARY_JSON}" ]] && ! is_truthy "${FORCE_NEW_SUMMARY}"; then
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${TS}" "${DIAG_JSON}" "${KNOWN_GOOD_B2_RUN}" "${CONC_LIST}" "${PRESSURE_POPULATE_CONC}" "${PRESSURE_THRASH_CONC}" "${BASELINE_REPLAY_CONC}" "${RESUME_FROM}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
sweep_concs = [int(x) for x in str(sys.argv[5]).split() if x.strip()]
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
        "slo_definition": "replay_p95_ms <= (baseline_B2_replay_p95_ms_at_concurrency1 * 1.10)",
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
obj.setdefault("baseline_b2_replay_p95_ms_at_concurrency1", None)
obj.setdefault("slo_replay_p95_ms", None)
obj.setdefault("max_concurrency_meeting_slo", {"B1": None, "B2": None})
path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
PY
else
  "${PYTHON_BIN}" - "${SWEEP_SUMMARY_JSON}" "${TS}" "${DIAG_JSON}" "${KNOWN_GOOD_B2_RUN}" "${CONC_LIST}" "${PRESSURE_POPULATE_CONC}" "${PRESSURE_THRASH_CONC}" "${BASELINE_REPLAY_CONC}" "${RESUME_FROM}" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
sweep_concs = [int(x) for x in str(sys.argv[5]).split() if x.strip()]
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
        "slo_definition": "replay_p95_ms <= (baseline_B2_replay_p95_ms_at_concurrency1 * 1.10)",
        "diagnosis_json": sys.argv[3],
        "known_good_b2_run": sys.argv[4],
        "resume_from": sys.argv[9] or None,
    },
    "rows": [],
    "status": "running",
    "decision_grade": None,
    "baseline_b2_replay_p95_ms_at_concurrency1": None,
    "slo_replay_p95_ms": None,
    "max_concurrency_meeting_slo": {"B1": None, "B2": None},
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
fi

RESUME_FROM_NORMALIZED=""
resume_gate_open="true"
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ "${RESUME_FROM}" =~ ^([Bb][12])_[Cc]([0-9]+)$ ]]; then
    RESUME_FROM_NORMALIZED="${BASH_REMATCH[1]^^}_c${BASH_REMATCH[2]}"
    resume_gate_open="false"
  else
    echo "Invalid BENCH_PHASE60_RESUME_FROM='${RESUME_FROM}'. Expected like B2_c2." >&2
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
      b2_req_hash="$("${PYTHON_BIN}" - "${b2_metrics_json}" <<'PY'
import json,sys
print((json.loads(sys.argv[1]).get("manifest_hashes") or {}).get("request_manifest_sha256",""))
PY
)"
      b2_prompts_hash="$("${PYTHON_BIN}" - "${b2_metrics_json}" <<'PY'
import json,sys
print((json.loads(sys.argv[1]).get("manifest_hashes") or {}).get("prompts_manifest_sha256",""))
PY
)"
      if [[ "${b2_req_hash}" != "${KNOWN_REQ_HASH}" || "${b2_prompts_hash}" != "${KNOWN_PROMPTS_HASH}" ]]; then
        b2_invalid_hash_row="$("${PYTHON_BIN}" - "${b2_metrics_json}" "${conc}" "${b2_cache}" "${KNOWN_REQ_HASH}" "${KNOWN_PROMPTS_HASH}" <<'PY'
import json,sys
m=json.loads(sys.argv[1])
m["mode"]="B2"
m["concurrency"]=int(sys.argv[2])
m["replay_concurrency"]=int(sys.argv[2])
m["status"]="invalid_manifest_hash"
m["phase"]="sweep_point"
m["expected_known_good_hashes"]={"request_manifest_sha256":sys.argv[4],"prompts_manifest_sha256":sys.argv[5]}
m["disk_offload_filter_disabled"]=False
m["kvbm_cache_dir"]=sys.argv[3]
print(json.dumps(m,separators=(",",":")))
PY
)"
        summary_upsert_row "${b2_invalid_hash_row}" "${point_key_b2}"
        update_summary_stop "baseline_manifest_hash_mismatch" "B2@c1 manifest hash mismatch vs known-good" "B2" "1"
        emit_stop_verdict "baseline_manifest_hash_mismatch" "B2@c1 manifest hash mismatch vs known-good" "B2" "1"
        exit 2
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
done

finalize_sweep_success

echo "DIAGNOSIS_JSON=${DIAG_JSON}"
echo "SWEEP_SUMMARY_JSON=${SWEEP_SUMMARY_JSON}"
echo "STOP_VERDICT_JSON=${STOP_VERDICT_JSON}"
echo "LOG_DIR=${LOG_DIR}"
