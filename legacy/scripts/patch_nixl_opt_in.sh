#!/usr/bin/env bash
set -euo pipefail

PO="/usr/local/lib/python3.12/dist-packages/dynamo/trtllm/utils/trtllm_utils.py"
PM="/usr/local/lib/python3.12/dist-packages/dynamo/trtllm/main.py"

python - <<'PY'
from pathlib import Path

po = Path("/usr/local/lib/python3.12/dist-packages/dynamo/trtllm/utils/trtllm_utils.py")
pm = Path("/usr/local/lib/python3.12/dist-packages/dynamo/trtllm/main.py")

# --- utils: add flag, default False, bind to config ---
txt = po.read_text()
if "use_nixl_connect" not in txt:
    txt = txt.replace(
        "self.publish_events_and_metrics: bool = publish_events_and_metrics",
        "self.publish_events_and_metrics: bool = publish_events_and_metrics\n        self.use_nixl_connect: bool = False",
    )
    txt = txt.replace(
        'parser.add_argument("--publish-events-and-metrics", action="store_true"',
        'parser.add_argument("--publish-events-and-metrics", action="store_true"\n    )\n    parser.add_argument("--use-nixl-connect", action="store_true", default=False, help="Enable NIXL Connect (default off)")',
    )
    txt = txt.replace(
        "config.migration_limit = args.migration_limit",
        "config.migration_limit = args.migration_limit\n    config.use_nixl_connect = args.use_nixl_connect",
    )
    po.write_text(txt)

# --- main: guard connector init ---
txt = pm.read_text()
if "NIXL Connect disabled via --use-nixl-connect" not in txt:
    txt = txt.replace(
"""    connector = None
    logging.info("Initializing NIXL Connect.")
    connector = nixl_connect.Connector()
    await connector.initialize()
""",
"""    connector = None
    if getattr(config, "use_nixl_connect", False):
        logging.info("Initializing NIXL Connect.")
        connector = nixl_connect.Connector()
        await connector.initialize()
    else:
        logging.info("NIXL Connect disabled via --use-nixl-connect")
""")
    pm.write_text(txt)

print("NIXL opt in flag patched")
PY
