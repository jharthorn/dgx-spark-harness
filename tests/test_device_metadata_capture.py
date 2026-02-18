import json
import unittest
from copy import deepcopy
from typing import Any, Optional
from unittest.mock import patch

from bench import run_bench


def _ok(stdout: str = "", stderr: str = "") -> dict[str, Any]:
    return {
        "success": True,
        "return_code": 0,
        "stdout": stdout,
        "stderr": stderr,
    }


def _fail(error: str, *, return_code: Optional[int] = 1, stdout: str = "", stderr: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error": error,
        "stdout": stdout,
        "stderr": stderr or error,
    }
    if return_code is not None:
        payload["return_code"] = return_code
    return payload


def _mock_runner(table: dict[tuple[str, ...], dict[str, Any]]):
    def _runner(cmd: list[str], timeout_s: float = 10.0) -> dict[str, Any]:  # noqa: ARG001
        key = tuple(cmd)
        base = {"timestamp_utc": "2026-02-12T00:00:00Z", "command": list(cmd)}
        if key not in table:
            base.update(_fail("command_not_mocked", return_code=127))
            return base
        base.update(table[key])
        return base

    return _runner


def _base_controller_fixture() -> dict[str, dict[str, Any]]:
    return {
        "nvme0": {
            "controller": "nvme0",
            "device": "/dev/nvme0",
            "namespace": "/dev/nvme0n1",
            "namespace_candidates": ["/dev/nvme0n1"],
            "bdf": "0000:5e:00.0",
            "sysfs_identity": {
                "model": "SYSFS_MODEL",
                "serial": "SYSFS_SERIAL",
                "firmware_rev": "SYSFS_FW",
            },
            "nvme_list_entries": [],
        }
    }


class DeviceMetadataCaptureTests(unittest.TestCase):
    def test_collect_device_metadata_success_schema(self) -> None:
        lsblk_payload = {
            "blockdevices": [
                {
                    "name": "nvme0n1",
                    "kname": "nvme0n1",
                    "path": "/dev/nvme0n1",
                    "model": "MOCK_MODEL",
                    "serial": "MOCK_SERIAL",
                    "size": "1.8T",
                    "rota": False,
                    "type": "disk",
                    "mountpoint": None,
                    "fstype": None,
                    "uuid": None,
                    "partuuid": None,
                }
            ]
        }
        command_table = {
            ("nvme", "list", "-o", "json"): _ok(
                json.dumps(
                    {
                        "Devices": [
                            {
                                "DevicePath": "/dev/nvme0n1",
                                "ModelNumber": "MOCK_MODEL",
                                "SerialNumber": "MOCK_SERIAL",
                                "Firmware": "MOCK_FW",
                                "PhysicalSize": 2000000000,
                            }
                        ]
                    }
                )
            ),
            ("nvme", "list-subsys", "-o", "json"): _ok(
                json.dumps(
                    {
                        "Subsystems": [
                            {
                                "Name": "nvme0",
                                "Paths": [{"Address": "0000:5e:00.0", "Transport": "pcie"}],
                            }
                        ]
                    }
                )
            ),
            ("lspci", "-D", "-nn"): _ok(
                "0000:5e:00.0 Non-Volatile memory controller: Mock NVMe Controller\n"
            ),
            ("findmnt", "-n", "-o", "SOURCE", "--target", "/mnt/nvme/kvbm"): _ok("/dev/nvme0n1p1\n"),
            (
                "lsblk",
                "-J",
                "-o",
                "NAME,KNAME,PATH,MODEL,SERIAL,SIZE,ROTA,TYPE,MOUNTPOINT,FSTYPE,UUID,PARTUUID",
            ): _ok(json.dumps(lsblk_payload)),
            ("findmnt", "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"): _ok(json.dumps({"filesystems": []})),
            ("mount",): _ok("/dev/nvme0n1 on /mnt/nvme type ext4 (rw,relatime)\n"),
            ("uname", "-a"): _ok("Linux mock-host 6.8.0 #1 SMP\n"),
            ("cat", "/etc/os-release"): _ok('NAME="MockOS"\nVERSION_ID="1"\n'),
            ("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"): _fail(
                "command_not_found",
                return_code=None,
            ),
            ("nvme", "id-ctrl", "/dev/nvme0", "--output-format=json"): _ok(
                json.dumps({"mn": "MOCK_MODEL", "sn": "MOCK_SERIAL", "fr": "MOCK_FW"})
            ),
            ("nvme", "id-ns", "/dev/nvme0n1", "--output-format=json"): _ok(json.dumps({"lbaf": 0, "flbas": 0})),
            ("nvme", "smart-log", "/dev/nvme0", "--output-format=json"): _ok(json.dumps({"temperature": 300})),
            ("lspci", "-vv", "-s", "0000:5e:00.0"): _ok(
                "LnkCap: Port #8, Speed 16GT/s, Width x4\nLnkSta: Speed 16GT/s, Width x4\n"
            ),
        }

        with patch("bench.run_bench._discover_nvme_controllers", side_effect=lambda: deepcopy(_base_controller_fixture())):
            with patch("bench.run_bench._run_capture_command", side_effect=_mock_runner(command_table)):
                payload = run_bench.collect_device_metadata(
                    capture_stage="pre",
                    nvme_device_hint="/dev/nvme0",
                    kvbm_cache_dir="/mnt/nvme/kvbm",
                    phase_io_device="nvme0n1",
                )

        self.assertEqual(payload["capture_stage"], "pre")
        self.assertIsInstance(payload.get("capture_timestamp"), str)
        self.assertEqual(payload.get("primary_nvme_device"), "/dev/nvme0")
        self.assertEqual(payload.get("primary_nvme_namespace"), "/dev/nvme0n1")
        self.assertEqual(payload.get("capture_errors"), [])
        self.assertEqual(len(payload.get("nvme_devices") or []), 1)
        self.assertEqual((payload.get("resolved_targets") or {}).get("primary_bdf"), "0000:5e:00.0")

        summary = payload.get("primary_storage_summary") or {}
        self.assertEqual(summary.get("model"), "MOCK_MODEL")
        self.assertEqual(summary.get("serial"), "MOCK_SERIAL")
        self.assertEqual(summary.get("firmware_rev"), "MOCK_FW")
        self.assertIn(summary.get("size"), (2000000000, "2000000000", "1.8T"))
        self.assertEqual(summary.get("pcie_link"), "16GT/s x4 (max 16GT/s x4)")

    def test_collect_device_metadata_fallback_and_errors(self) -> None:
        lsblk_payload = {
            "blockdevices": [
                {
                    "name": "nvme0n1",
                    "kname": "nvme0n1",
                    "path": "/dev/nvme0n1",
                    "model": "LSBLK_MODEL",
                    "serial": "LSBLK_SERIAL",
                    "size": "953.9G",
                    "rota": False,
                    "type": "disk",
                    "mountpoint": None,
                    "fstype": None,
                    "uuid": None,
                    "partuuid": None,
                }
            ]
        }
        command_table = {
            ("nvme", "list", "-o", "json"): _fail("command_not_found", return_code=None),
            ("nvme", "list-subsys", "-o", "json"): _fail("command_not_found", return_code=None),
            ("lspci", "-D", "-nn"): _fail("command_not_found", return_code=None),
            ("findmnt", "-n", "-o", "SOURCE", "--target", "/mnt/nvme/kvbm"): _fail("permission denied"),
            (
                "lsblk",
                "-J",
                "-o",
                "NAME,KNAME,PATH,MODEL,SERIAL,SIZE,ROTA,TYPE,MOUNTPOINT,FSTYPE,UUID,PARTUUID",
            ): _ok(json.dumps(lsblk_payload)),
            ("findmnt", "-J", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"): _ok(json.dumps({"filesystems": []})),
            ("mount",): _ok(""),
            ("uname", "-a"): _ok("Linux mock-host 6.8.0 #1 SMP\n"),
            ("cat", "/etc/os-release"): _ok('NAME="MockOS"\nVERSION_ID="1"\n'),
            ("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"): _fail(
                "command_not_found",
                return_code=None,
            ),
            ("nvme", "id-ctrl", "/dev/nvme0", "--output-format=json"): _fail("command_not_found", return_code=None),
            ("nvme", "id-ns", "/dev/nvme0n1", "--output-format=json"): _fail("command_not_found", return_code=None),
            ("nvme", "smart-log", "/dev/nvme0", "--output-format=json"): _fail("permission denied", return_code=2),
            ("smartctl", "-a", "-j", "/dev/nvme0"): _ok(json.dumps({"smart_status": {"passed": True}})),
            ("lspci", "-vv", "-s", "0000:5e:00.0"): _fail("command_not_found", return_code=None),
        }

        with patch("bench.run_bench._discover_nvme_controllers", side_effect=lambda: deepcopy(_base_controller_fixture())):
            with patch("bench.run_bench._run_capture_command", side_effect=_mock_runner(command_table)):
                payload = run_bench.collect_device_metadata(
                    capture_stage="post",
                    nvme_device_hint="/dev/nvme0",
                    kvbm_cache_dir="/mnt/nvme/kvbm",
                    phase_io_device="nvme0n1",
                )

        self.assertEqual(payload["capture_stage"], "post")
        self.assertEqual(payload.get("primary_nvme_device"), "/dev/nvme0")
        self.assertGreater(len(payload.get("capture_errors") or []), 0)

        summary = payload.get("primary_storage_summary") or {}
        self.assertEqual(summary.get("model"), "SYSFS_MODEL")
        self.assertEqual(summary.get("serial"), "SYSFS_SERIAL")
        self.assertEqual(summary.get("firmware_rev"), "SYSFS_FW")
        self.assertEqual(summary.get("size"), "953.9G")

        devices = ((payload.get("nvme") or {}).get("devices") or [])
        self.assertEqual(len(devices), 1)
        self.assertIsInstance(devices[0].get("smartctl"), dict)
        self.assertTrue((devices[0].get("smartctl") or {}).get("success"))

    def test_collect_device_metadata_safe_falls_back_on_internal_error(self) -> None:
        with patch("bench.run_bench.collect_device_metadata", side_effect=RuntimeError("boom")):
            payload = run_bench.collect_device_metadata_safe(
                capture_stage="pre",
                nvme_device_hint="/dev/nvme0",
                kvbm_cache_dir="/mnt/nvme/kvbm",
                phase_io_device="nvme0n1",
            )

        self.assertEqual(payload.get("capture_stage"), "pre")
        self.assertIsInstance(payload.get("capture_timestamp"), str)
        self.assertIsInstance(payload.get("capture_errors"), list)
        self.assertGreater(len(payload.get("capture_errors") or []), 0)
        self.assertEqual(payload.get("primary_nvme_device"), "/dev/nvme0")
        self.assertIsInstance((payload.get("nvme") or {}).get("list"), dict)
        self.assertIsInstance((payload.get("block_filesystem") or {}).get("lsblk"), dict)


if __name__ == "__main__":
    unittest.main()
