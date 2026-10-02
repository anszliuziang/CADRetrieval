"""Server entry: Text2CAD G4 full-matrix two-card launch controller.

Thin wrapper around ``idea01.text2cad_g4_launch.main``; ALL verification,
fixed-command generation, common-deadline and evidence logic lives in the
module (reviewed there). The controller itself performs NO model/BERT/NPU
forward -- it only spawns the two audited scorer processes after every
pre-launch gate passes, and it is NOT authorized to run as of 2026-10-02
(the Codex-approved full SHA must be supplied via --approved-sha after code
review, and origin/master ACTIVE_TASK must state pilot_authorized with the
fixed G4 task/run/budget scope).

Production usage (yunnan-share2, AFTER Codex approval; NOT authorized now):

    nohup /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
      /home/ma-user/work/CADRetrieval/experiments/scripts/run_text2cad_g4_launch.py \
      --approved-sha <40-hex approved by Codex> \
      > /home/ma-user/work/CADRetrieval_runtime/runs/g4_controller.nohup.log 2>&1 &

The controller writes every artifact under
$RT/runs/idea01_text2cad_g4_full_20261002/ (created by itself, fail-if-exists).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_g4_launch import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
