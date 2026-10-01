"""Server entry: frozen Text2CAD full-matrix scorer (M1c B2).

Thin wrapper around ``idea01.text2cad_frozen_runner.main``; ALL gates live in
the module (reviewed pair-manifest fingerprints -> freshly fetched
origin/master ACTIVE_TASK authorization -> committed 172-key manifest ->
shared two-card budget ledger -> strict zero-drift load). ``--mode dry-run``
is the only forward-free audit and consumes no budget.

Production template (yunnan-share2, after Codex approval; NOT authorized as
of 2026-09-30):

    PYTHONDONTWRITEBYTECODE=1 HF_HOME=$RUNTIME/hf XDG_CACHE_HOME=$RUNTIME/xdg \
    TMPDIR=$RUNTIME/tmp \
    /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
      experiments/scripts/run_text2cad_score_matrix.py \
      --mode score \
      --checkpoint $RT/models/Text2CAD-v1.0/422d4c93.../Text2CAD_1.0.pth \
      --bert-dir $RT/models/bert-large-uncased/6da4b6a2.../ \
      --cad-seq-zip $RT/datasets/Text2CAD-v1.0/422d4c93.../cad_seq.zip \
      --pair-manifest-dir $RT/pair_manifest_v3 \
      --expected-keys-manifest experiments/configs/text2cad_checkpoint_expected_keys_20260930.json \
      --out-dir $RT/runs/<run_id>/shard0 --device npu:0 --batch-size 8 \
      --shard-id 0 --n-shards 2 --cards-used 1 --repo-root /home/ma-user/work/CADRetrieval

The second process uses shard1, npu:1, shard-id 1 and its own out-dir.

where $RT=/home/ma-user/work/CADRetrieval_runtime (checkout OUTSIDE paths).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_frozen_runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
