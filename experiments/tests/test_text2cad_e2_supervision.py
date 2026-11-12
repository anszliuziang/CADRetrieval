"""Standalone POSIX supervision regressions for the E2 deadline supervisor
(review round 7, item 3): NO model, NO G4 assets, NO score entry — a real
child process that IGNORES SIGTERM is supervised via the real ledger file,
and the regressions assert the exit SIGNAL (SIGKILL) and the timing against
the absolute schedule (grace counted INTO the caps). Skips on Windows.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01 import text2cad_e2_runner as e2mod  # noqa: E402

CHILD_CODE = (
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"  # a stuck scorer cannot run handlers
    "t0 = time.time()\n"
    "while time.time() - t0 < 600:\n"
    "    time.sleep(0.5)\n"
)


def _write_pending_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "invocations": [{"invocation_id": "SUP", "finalized": False,
                         "started_epoch": time.time()}],
        "card_seconds_spent": 0.0,
    }), encoding="utf-8")


@unittest.skipUnless(os.name == "posix" and hasattr(os, "fork"),
                     "POSIX process supervision: runs on the Linux preflight host")
class TestSupervisionProcess(unittest.TestCase):
    """Supervision is exercised through the REAL _supervise_loop in a forked
    process targeting a REAL child process; the child ignores SIGTERM so only
    the SIGKILL at the absolute hard deadline can end it."""

    def _run_case(self, *, wall_deadline, max_card_seconds):
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        ledger = Path(td.name) / "led.json"
        _write_pending_ledger(ledger)

        child = subprocess.Popen(
            [sys.executable, "-c", CHILD_CODE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        sup = ctx.Process(target=e2mod._supervise_loop,
                          args=(child.pid, str(ledger), wall_deadline,
                                max_card_seconds, 1),
                          daemon=True)
        sup.start()
        t_start = time.time()
        sup.join(timeout=max(wall_deadline - t_start + 60, 60))

        deadline_abs = min(wall_deadline, t_start + max_card_seconds)
        status = None
        while time.time() < deadline_abs + 5.0:
            w, st = os.waitpid(child.pid, os.WNOHANG)
            if w != 0:
                status = st
                break
            time.sleep(0.1)
        died_at = time.time()
        self.assertIsNot(status, None, "child survived past hard+5s")
        self.assertTrue(os.WIFSIGNALED(status),
                        f"child must die by SIGKILL, status={status}")
        self.assertEqual(os.WTERMSIG(status), 9, "the killing signal must be SIGKILL")
        hard = min(wall_deadline, t_start + max_card_seconds)  # card arm: rate=1/s
        self.assertGreaterEqual(died_at, hard - 2.0, "KILL before the schedule")
        self.assertLessEqual(died_at, hard + 2.5, "KILL missed the absolute deadline")
        if sup.is_alive():
            sup.terminate()
        return died_at - t_start

    def test_card_arm_kills_by_card_deadline(self):
        from idea01 import text2cad_e2_runner as e2mod

        # cap 5 card-seconds at 1 card-second/second -> hard = snapshot + 5;
        # TERM would be at hard-30 (already past) -> immediate TERM, KILL at +5
        elapsed = self._run_case(wall_deadline=time.time() + 600,
                                 max_card_seconds=5.0)
        self.assertLess(elapsed, 12.0, "card arm must fire promptly (~5s)")

    def test_wall_arm_kills_by_wall_deadline(self):
        from idea01 import text2cad_e2_runner as e2mod

        # wall deadline +32s -> TERM at +2, child ignores it, KILL at +32
        elapsed = self._run_case(wall_deadline=time.time() + 32.0,
                                 max_card_seconds=1e9)
        self.assertGreaterEqual(elapsed, 30.0, "KILL before the wall deadline")
        self.assertLessEqual(elapsed, 34.0, "KILL missed the wall deadline")


if __name__ == "__main__":
    unittest.main()
