# Phase 1 Harness Checkpoint (2026-02)

## Scope
Checkpoint before higher-footprint tuning, 70B expansion, and Dataset L context-ladder work.

## What We Shipped

- Phase60 fixed-pressure sweep guardrails:
  - semantic baseline hash policy (`strict`/`warn`/`accept`) with known-good hash file and append-only audit log,
  - enforced B1 disk-tier-off policy with explicit `B1_DISK_TIER_NOT_DISABLED` invalidation semantics,
  - explicit metric policy in sweep artifacts (`ttfc_ms` preferred, `ttft_ms` fallback when TTFC absent).
- Phase60 post-processing:
  - compact knee-table extractor (`scripts/phase60_extract_knee_table.py`) -> `phase60_knee_table_<ts>.csv`,
  - crossover recommender (`scripts/recommend_phase70_crossover_from_phase60.py`) with JSON/Markdown outputs and rejection/warning diagnostics.
- Phase70 publish-grade flow:
  - canonical runner supports replay concurrency `c>=1` and propagates `replay_concurrency` labels through analyzer and pack outputs,
  - preflight fast-fail gate (`scripts/bench_phase70_preflight.sh`) with explicit reason codes,
  - early mechanism gate (`scripts/phase70_check_mechanism_gate.py`) after first B2 leg,
  - post-run verdict writer (`scripts/phase70_write_verdict.py`) with explicit write/rehydrate/reuse semantics and decision-grade policy,
  - results pack integration (`scripts/make_phase70_results_pack.py`) now carries `verdict.json` into publish packs.
- Canonical workflow docs aligned in:
  - `README.md`
  - `RUNBOOK.md`
  - `bench/README.md`

## Canonical Phase 1 Operator Loop

1. Run Phase60 sweep (`scripts/bench_phase60_rehydrate_minimal_sweep.sh`).
2. Extract knee table (`scripts/phase60_extract_knee_table.py`).
3. Generate crossover recommendation (`scripts/recommend_phase70_crossover_from_phase60.py`).
4. Run Phase70 paired repeats (`scripts/bench_phase70_rehydrate_pair_repeats.sh`).
5. Build publish pack (`scripts/make_phase70_results_pack.py`).

## What We Validated

### Unit Tests

Command run:

```bash
python3 -m unittest tests.test_phase60_baseline_manifest_semantic tests.test_phase60_extract_knee_table tests.test_recommend_phase70_crossover_from_phase60 tests.test_phase70_check_mechanism_gate tests.test_phase70_write_verdict tests.test_analyze_phase70_pairs tests.test_make_phase70_results_pack
```

Result:

- `Ran 20 tests in 0.025s`
- `OK`

### Script Parse Validation

Command run:

```bash
bash -n scripts/bench_phase60_rehydrate_minimal_sweep.sh scripts/bench_phase70_preflight.sh scripts/bench_phase70_rehydrate_c1_pair_repeats.sh scripts/bench_phase70_rehydrate_pair_repeats.sh
```

Result:

- no parse errors

### Representative Existing Artifact Sets (TS)

- Phase60 sweep + knee + recommendation:
  - `20260219T211627Z`
  - includes summary JSON/CSV, knee CSV, recommendation JSON/MD.
- Phase70 paired repeats + publish pack:
  - `20260219T161922Z`
  - includes manifest/summary/deltas/order-check/verdict and publish pack `phase70_pairs6_c4_20260219T161922Z`.

## Known Limitations

- TTFC is still absent in some run families; tooling falls back to TTFT and labels fallback usage.
- Many local engines are still capped at `max_num_tokens=8192`; Dataset L (`>=32k`) remains blocked until context-capable engine rebuilds are available.

## Next-Phase Plan

- Increase footprint pressure knobs (prefix/session/pressure/replay repeats) to strengthen effect-size separation.
- Expand publish-grade runs on 70B profiles with the same guardrailed loop.
- Execute Dataset L context ladder (`16k -> 32k -> higher`) with explicit rebuild provenance and gates.

