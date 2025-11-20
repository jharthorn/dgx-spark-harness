#!/usr/bin/env bash
set -euo pipefail
# Patch installed dynamo.trtllm in a container so --use-nixl-connect is opt-in (default False).
# Applies a guard around the connector init and fixes argparse to store_true.

PO="/usr/local/lib/python3.12/dist-packages/dynamo/trtllm/utils/trtllm_utils.py"
PM="/usr/local/lib/python3.12/dist-packages/dynamo/trtllm/main.py"

python3 - <<'PY'
from pathlib import Path
po = Path("${PO}")
txt = po.read_text()
txt = txt.replace(
    "self.publish_events_and_metrics: bool = False\n        self.disaggregation_mode",
    "self.publish_events_and_metrics: bool = False\n        self.use_nixl_connect: bool = False\n        self.disaggregation_mode",
)
txt = txt.replace(
    'max_beam_width={self.max_beam_width},\n             f"free_gpu_memory_fraction={self.free_gpu_memory_fraction}, "\n             f"extra_engine_args',
    'max_beam_width={self.max_beam_width},\n             f"free_gpu_memory_fraction={self.free_gpu_memory_fraction}, "\n             f"use_nixl_connect={self.use_nixl_connect}, "\n             f"extra_engine_args',
)
txt = txt.replace(
    'parser.add_argument(\n        "--use-nixl-connect",\n        type=bool,\n        default=False,\n        help="Use NIXL Connect for communication between workers.",\n    )',
    'parser.add_argument(\n        "--use-nixl-connect",\n        action="store_true",\n        default=False,\n        help="Use NIXL Connect for communication between workers (off by default).",\n    )',
)
txt = txt.replace(
    "config.migration_limit = args.migration_limit\n    config.extra_engine_args = args.extra_engine_args",
    "config.migration_limit = args.migration_limit\n    config.use_nixl_connect = args.use_nixl_connect\n    config.extra_engine_args = args.extra_engine_args",
)
po.write_text(txt)
print("patched utils")
PY

python3 - <<'PY'
import re
from pathlib import Path
pm = Path("${PM}")
txt = pm.read_text()
new = (
    "    connector = None\n"
    "    if config.use_nixl_connect:\n"
    "        logging.info(\"Initializing NIXL Connect.\")\n"
    "        connector = nixl_connect.Connector()\n"
    "        await connector.initialize()\n"
    "    else:\n"
    "        logging.info(\"NIXL Connect disabled via --use-nixl-connect\")"
)
out, n = re.subn(
    r"    connector = None\s+"
    r"logging\\.info\\(\"Initializing NIXL Connect\\.\"\)\s+"
    r"connector = nixl_connect\\.Connector\\(\)\s+"
    r"await connector\\.initialize\\(\)",
    new,
    txt,
    flags=re.M,
)
if n == 0:
    raise SystemExit("patch failed; edit main.py manually around connector init")
pm.write_text(out)
print("patched main, replacements:", n)
PY

echo "NIXL opt-in patch applied"
