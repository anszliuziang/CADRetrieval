from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import continue_retrieval_benchmark_native_probe as continuation


def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


class NativeContinuationTests(unittest.TestCase):
    def test_waits_for_latest_first_batch_attempt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "control/finish_first_batch/current.json"
            current = root / "control/finish_first_batch_retry1/current.json"
            put(old, {"state": "completed", "utc_epoch": 100})
            put(current, {"state": "partial_external", "utc_epoch": 110})
            with patch("time.time", return_value=120):
                with self.assertRaisesRegex(RuntimeError, "unresolved acceptance"):
                    continuation.wait_for_first_batch(root, root / "continuation", 200)
                put(current, {"state": "completed", "utc_epoch": 120})
                self.assertEqual(continuation.wait_for_first_batch(root, root / "continuation", 200),
                                 current.parent)

    def test_ssh_gap_uses_recorded_connection_start(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root / "received/duoduoclip_full/connection_start.json", {"started_epoch": 115})
            put(root / "control/share4_npu_receipt.json", {"started_epoch": 125})
            clock = [140.0]

            def advance(seconds):
                clock[0] += seconds

            with patch("time.time", side_effect=lambda: clock[0]), patch("time.sleep", side_effect=advance):
                continuation.wait_for_ssh_gap(root, 130, 400)
            self.assertEqual(clock[0], 195.0)

    def test_remote_launch_uses_separate_git_ref_and_output(self):
        command = continuation.remote_command("a" * 40, "b" * 64, "c" * 64, 2000)
        self.assertLess(command.index("controller.actual_exit"), command.index("git clone"))
        self.assertLess(command.index('"$run/full/ledger.json"'), command.index("git clone"))
        self.assertIn('git -C "$checkout" fetch --no-tags', command)
        self.assertNotIn('git -C "$seed" fetch', command)
        self.assertIn('out="$run/probe_native_tail"', command)
        self.assertIn("--mode probe", command)
        self.assertNotIn("--mode score", command)

    def test_armed_duoduo_collector_goes_first_at_the_same_connection_window(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            put(root / "control/duoduo_full_collect_launcher.json", {"local_pid": 1})
            clock = [165.0]

            def advance(seconds):
                clock[0] += seconds
                if clock[0] >= 200:
                    put(root / "received/duoduoclip_full/connection_start.json", {"started_epoch": 200})

            with patch("time.time", side_effect=lambda: clock[0]), patch("time.sleep", side_effect=advance):
                continuation.wait_for_ssh_gap(root, 100, 400)
            self.assertEqual(clock[0], 265.0)

    def test_probe_verification_checks_settlement_and_long_text(self):
        with TemporaryDirectory() as tmp:
            raw = Path(tmp)
            probe = raw / continuation.OUTPUT_DIRECTORY
            status = raw / "control/native_tail_probe/controller.actual_exit"
            status.parent.mkdir(parents=True)
            status.write_text("0\n", encoding="ascii")
            auth_hash = "a" * 40
            auth = {"deadline_epoch": 250, "freeze_sha256": "b" * 64, "replay_sha256": "c" * 64}
            put(probe / "run_authorization.json", auth)
            put(probe / "controller_exit.json", {"exit_code": 0, "conditions": ["probe"],
                "authorization_sha": auth_hash, "implementation_sha": continuation.SCORER})
            put(probe / "probe_exit_200.json", {"exits": [0]})
            invocation = {"finalized": True, "exit_code": 0, "cards_used": 1, "card_seconds": 60,
                          "pairs_charged": 2560, "started_epoch": 140, "ended_epoch": 200}
            ledger = {"approved_sha": continuation.SCORER, "pairs_charged": 2560, "max_pairs": 2560,
                      "max_card_seconds": 900, "card_seconds_spent": 60, "invocations": [invocation]}
            put(probe / "ledger.json", ledger)
            cases = []
            for arm in ["l0_history"] * 4 + ["l1"] * 3 + ["l3"] * 3:
                count = 600 if arm == "l3" else 40
                cases.append({"arm": arm, "pairs_per_pass": 128, "cache_bitwise_equal": True,
                              "cached_batch_seconds": [0.03] * 8, "uncached_batch_seconds": [0.06] * 8,
                              "text_tokens": {"token_count": count, "effective_token_count": min(count, 512),
                                              "truncated": count > 512}})
            report = {"passed": True, "implementation_sha": continuation.SCORER, "conditions": ["l1", "l3"],
                      "freeze_sha256": auth["freeze_sha256"], "replay_sha256": auth["replay_sha256"],
                      "pairs_charged": 2560, "cache_passed": True, "cache_hits": 70, "measurements": cases,
                      "asset_preparation_seconds": 12, "peak_memory_allocated_bytes": 100,
                      "peak_memory_reserved_bytes": 200}
            put(probe / "probe_result.json", report)
            accepted = continuation.verify_probe(raw, auth_hash, auth)
            self.assertEqual((accepted["pairs"], accepted["card_seconds"]), (2560, 60))
            self.assertEqual(len(accepted["projections"]), 4)
            invocation["ended_epoch"] = 251
            put(probe / "ledger.json", ledger)
            with self.assertRaisesRegex(ValueError, "settlement"):
                continuation.verify_probe(raw, auth_hash, auth)
            invocation["ended_epoch"] = 200
            put(probe / "ledger.json", ledger)
            for row in cases[-3:]:
                row["text_tokens"] = {"token_count": 400, "effective_token_count": 400, "truncated": False}
            put(probe / "probe_result.json", report)
            with self.assertRaisesRegex(ValueError, "overlength representative"):
                continuation.verify_probe(raw, auth_hash, auth)


if __name__ == "__main__":
    unittest.main()
