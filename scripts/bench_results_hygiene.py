#!/usr/bin/env python3
"""Archive low-signal result artifacts and keep a compact modern root layout.

Default behavior is dry-run. Use --execute to apply moves.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

# Modern Phase60/70 artifact naming expected for current methodology.
MODERN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^phase60_rehydrate_minimal_preflight_B2_c1_\d{8}T\d{6}Z$"),
    re.compile(r"^phase60_rehydrate_minimal_sweep_B[0-2]_c\d+_\d{8}T\d{6}Z$"),
    re.compile(r"^phase60_minimal_sweep_logs_\d{8}T\d{6}Z$"),
    re.compile(r"^phase60_rehydrate_concurrency_sweep_summary_minimal_\d{8}T\d{6}Z\.(json|csv)$"),
    re.compile(r"^phase60_sweep_b2c1_failure_diagnosis_\d{8}T\d{6}Z\.json$"),
    re.compile(r"^phase60_matrix_stop_verdict_minimal_\d{8}T\d{6}Z\.json$"),
    re.compile(r"^phase60_stream_smoke_B2_c1_\d{8}T\d{6}Z$"),
    re.compile(r"^phase70_rehydrate_pair_B[0-2]_p\d+_l\d+_\d{8}T\d{6}Z$"),
    re.compile(r"^phase70_pair_logs_\d{8}T\d{6}Z$"),
    re.compile(r"^phase70_publishability_watch_\d{8}T\d{6}Z\.log$"),
    re.compile(r"^phase70_rehydrate_pair_repeats_(manifest|summary|deltas|order_check)_\d{8}T\d{6}Z\.(json|csv)$"),
)

# Trial-like artifacts are separated from general legacy to make root high-signal.
TRIAL_KEYWORDS = (
    "trial",
    "smoke",
    "debug",
    "dry",
    "retry",
    "probe",
    "validation",
    "verify",
    "diag",
    "diagnosis",
    "watch",
    "selftest",
    "runtime_compat",
    "control_plane",
    "compare_",
    "live_pair",
    "handoff",
)


@dataclass
class EntryDecision:
    name: str
    action: str  # keep | move_trial | move_legacy
    reason: str
    source: Path
    destination: Path | None = None


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact bench/results into modern high-signal layout.")
    parser.add_argument("--results-root", default="bench/results", help="Results root directory.")
    parser.add_argument(
        "--keep-latest-ts",
        type=int,
        default=1,
        help="Keep modern artifacts for the latest N timestamps.",
    )
    parser.add_argument(
        "--keep-modern-ts",
        action="append",
        default=[],
        help="Explicit modern timestamp to keep (can be repeated).",
    )
    parser.add_argument(
        "--archive-tag",
        default=None,
        help="Archive bucket tag (default: current UTC compact timestamp).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply moves. Without this flag, only prints a plan.",
    )
    return parser.parse_args()


def is_modern(name: str) -> bool:
    return any(pattern.match(name) for pattern in MODERN_PATTERNS)


def extract_ts(name: str) -> str | None:
    match = TS_RE.search(name)
    if not match:
        return None
    return match.group(1)


def is_trial_like(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in TRIAL_KEYWORDS)


def iter_entries(results_root: Path) -> Iterable[Path]:
    for path in sorted(results_root.iterdir(), key=lambda item: item.name):
        if path.name in {"archive", "trials"}:
            continue
        yield path


def resolve_keep_timestamps(entries: list[Path], keep_latest: int, pinned: list[str]) -> set[str]:
    modern_ts: set[str] = set()
    for entry in entries:
        if not is_modern(entry.name):
            continue
        ts = extract_ts(entry.name)
        if ts:
            modern_ts.add(ts)
    newest = sorted(modern_ts, reverse=True)[: max(keep_latest, 0)]
    keep = set(newest)
    keep.update(ts for ts in pinned if ts)
    return keep


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}__dup{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def build_plan(results_root: Path, keep_ts: set[str], archive_tag: str) -> list[EntryDecision]:
    archive_legacy_root = results_root / "archive" / archive_tag
    archive_trial_root = results_root / "trials" / archive_tag

    plan: list[EntryDecision] = []
    for entry in iter_entries(results_root):
        name = entry.name
        if is_modern(name):
            ts = extract_ts(name)
            if ts and ts in keep_ts:
                plan.append(EntryDecision(name=name, action="keep", reason=f"modern_ts={ts}", source=entry))
                continue
            destination = archive_legacy_root / name
            plan.append(
                EntryDecision(
                    name=name,
                    action="move_legacy",
                    reason=f"modern_old_ts={ts or 'none'}",
                    source=entry,
                    destination=destination,
                )
            )
            continue

        if is_trial_like(name):
            destination = archive_trial_root / name
            plan.append(
                EntryDecision(
                    name=name,
                    action="move_trial",
                    reason="trial_keyword",
                    source=entry,
                    destination=destination,
                )
            )
            continue

        destination = archive_legacy_root / name
        plan.append(
            EntryDecision(
                name=name,
                action="move_legacy",
                reason="non_modern",
                source=entry,
                destination=destination,
            )
        )
    return plan


def apply_plan(plan: list[EntryDecision]) -> None:
    for decision in plan:
        if decision.action == "keep" or decision.destination is None:
            continue
        destination = unique_destination(decision.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(decision.source), str(destination))
        decision.destination = destination


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists() or not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")

    archive_tag = args.archive_tag or now_ts()
    entries = list(iter_entries(results_root))
    keep_ts = resolve_keep_timestamps(entries, args.keep_latest_ts, list(args.keep_modern_ts))
    plan = build_plan(results_root, keep_ts, archive_tag)

    keep_count = sum(1 for item in plan if item.action == "keep")
    trial_count = sum(1 for item in plan if item.action == "move_trial")
    legacy_count = sum(1 for item in plan if item.action == "move_legacy")

    print(f"results_root={results_root}")
    print(f"archive_tag={archive_tag}")
    print(f"keep_timestamps={sorted(keep_ts)}")
    print(f"plan_counts keep={keep_count} trial_moves={trial_count} legacy_moves={legacy_count}")
    for item in plan:
        if item.action == "keep":
            print(f"KEEP {item.name} ({item.reason})")
        else:
            print(f"MOVE {item.name} -> {item.destination} ({item.reason})")

    if not args.execute:
        print("dry_run=true (use --execute to apply)")
        return 0

    apply_plan(plan)
    print("dry_run=false applied=true")
    print(
        f"archive_paths: legacy={results_root / 'archive' / archive_tag} "
        f"trials={results_root / 'trials' / archive_tag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
