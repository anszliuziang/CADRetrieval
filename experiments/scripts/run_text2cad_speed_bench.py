"""Server entry: Text2CAD G1/G2 benchmark + optimization gates (M1c B2/G2 repair).

Thin wrapper around ``idea01.text2cad_speed_bench.main``. Pure-CPU modes
(``select``, ``eta``, ``eta-g2``, ``audit-text-lengths``, ``select-g2``) run
anywhere; forward-consuming modes (``spot``, ``batch-consistency``,
``trim-check``, ``cache-check``, ``throughput``, ``profile-consistency``)
share the matrix scorer's fail-closed authorization + shared budget ledger
and run only on the approved server checkout. The G2-repair modes pin the
COMMITTED G2 selection config
(``experiments/configs/text2cad_g2_pair_selection_20261001.json``), which
does not exist yet -- they fail closed until it is generated from the audited
real BERT token lengths, reviewed and committed, and a full SHA is
authorized. NOTHING in this round authorizes a real forward.

Production template (after Codex approval; NOT authorized as of 2026-09-30):

    PYTHONDONTWRITEBYTECODE=1 HF_HOME=$RUNTIME/hf XDG_CACHE_HOME=$RUNTIME/xdg \
    TMPDIR=$RUNTIME/tmp \
    /home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python \
      experiments/scripts/run_text2cad_speed_bench.py \
      --mode spot \
      --checkpoint $RT/models/Text2CAD-v1.0/422d4c93.../Text2CAD_1.0.pth \
      --bert-dir $RT/models/bert-large-uncased/6da4b6a2.../ \
      --cad-seq-zip $RT/datasets/Text2CAD-v1.0/422d4c93.../cad_seq.zip \
      --pair-manifest-dir $RT/pair_manifest_v1 \
      --expected-keys-manifest experiments/configs/text2cad_checkpoint_expected_keys_20260930.json \
      --selection experiments/configs/text2cad_g1g2_pair_selection_20260930.json \
      --out-dir $RT/runs/<bench_id> --device npu:0 --cards-used 1 \
      --repo-root /home/ma-user/work/CADRetrieval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_speed_bench import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
