"""Standalone POSIX supervision regressions for the E2 deadline supervisor
(review round 9): NO model, NO G4 assets, NO score entry — a REAL child
process that IGNORES SIGTERM (a stuck scorer cannot run handlers) is
supervised via the real ledger file, and the regressions assert the exit
SIGNAL (SIGKILL) and the death timing against the ABSOLUTE schedule for both
the card arm and the wall arm. Skips on Windows; executes on the share4
Linux preflight host.
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

# the child installs SIG_IGN as the VERY first statement (a stuck scorer
# cannot run handlers — only SIGKILL at the absolute hard deadline may end
# it), then writes a ready marker so the test never races the startup
CHILD_CODE = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "open(sys.argv[1], 'w').write('ready')\n"
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

    def _spawn_child_and_supervisor(self, *, wall_deadline, max_card_seconds):
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        ledger = Path(td.name) / "led.json"
        marker = Path(td.name) / "child_ready"

        # start the child and wait until SIG_IGN is installed — otherwise the
        # supervisor's TERM could win a startup race and kill the child
        # before it ignores anything (making the KILL-timing assert moot)
        child = subprocess.Popen(
            [sys.executable, "-c", CHILD_CODE, str(marker)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        t_wait0 = time.time()
        while not marker.exists() and time.time() - t_wait0 < 10.0:
            time.sleep(0.05)
        self.assertTrue(marker.exists(), "child never installed SIG_IGN")
        self.addCleanup(child.kill)

        _write_pending_ledger(ledger)
        sup = ctx.Process(target=e2mod._supervise_loop,
                          args=(child.pid, str(ledger), wall_deadline,
                                max_card_seconds, 1),
                          daemon=True)
        sup.start()
        t_start = time.time()
        sup.join(timeout=max(wall_deadline - t_start + 60, 60))

        hard = min(wall_deadline, t_start + max_card_seconds)
        status = None
        while time.time() < hard + 5.0:
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
        return died_at, hard

    def test_card_arm_kills_by_card_deadline(self):
        # cap 5 card-seconds at 1/s -> hard = snapshot + 5; TERM fires at
        # hard-30 (already past) -> immediate TERM, ignored by the child;
        # SIGKILL lands at hard
        died_at, hard = self._spawn_child_and_supervisor(
            wall_deadline=time.time() + 600, max_card_seconds=5.0)
        self.assertGreaterEqual(died_at, hard - 2.0, "KILL before the schedule")
        self.assertLessEqual(died_at, hard + 2.5, "KILL missed the absolute deadline")
        self.assertLess(died_at - (hard - 5.0), 12.0, "card arm must fire promptly (~5s)")

    def test_wall_arm_kills_by_wall_deadline(self):
        # wall deadline +32s -> TERM at +2 (ignored by the child), KILL at +32
        # (the helper asserts |died_at - hard| <= 2.5 against the ABSOLUTE hard)
        died_at, hard = self._spawn_child_and_supervisor(
            wall_deadline=time.time() + 32.0, max_card_seconds=1e9)
        self.assertGreaterEqual(died_at, 30.0, "KILL before the wall deadline")
        self.assertLessEqual(abs(died_at - hard), 2.5)


if __name__ == "__main__":
    unittest.main()
