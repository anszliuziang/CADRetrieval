#!/usr/bin/env python
"""E2 background scoring entry (single card; plan IDEA01_E2_EMPTY_TEXT_BACKGROUND_2026-11-12.md).

    # CPU dry-run preflight (zero forward; identical input validation as score):
    python experiments/scripts/run_text2cad_e2_background.py --mode dry-run \
      --bert-dir $RT/models/bert-large-uncased/6da4b6a2... \
      --cad-seq-zip $RT/datasets/Text2CAD-v1.0/422d4c93.../cad_seq.zip \
      --b0-manifest-dir $RT/text2cad_e2_manifests_20261112/b0 \
      --b1-manifest-dir $RT/text2cad_e2_manifests_20261112/b1 \
      --replay-manifest-dir $RT/text2cad_e2_manifests_20261112/replay \
      --original-manifest-dir $RT/pair_manifest_v4 \
      --out-dir $RT/runs/idea01_text2cad_e2_background_20261112/dry_run

    # score (REQUIRES user-approved launch: ACTIVE_TASK e2_* fields on the
    # pinned authorization ref, detached clean HEAD at e2_approved_sha):
    python experiments/scripts/run_text2cad_e2_background.py --mode score \
      --which b0 --device npu:0 \
      --checkpoint $RT/models/Text2CAD-v1.0/422d4c93.../Text2CAD_1.0.pth \
      (same manifest/out-dir arguments) [--repo-root /home/ma-user/work/CADRetrieval]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_e2_runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
