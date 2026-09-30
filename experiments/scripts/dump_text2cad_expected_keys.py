"""Dump the expected non-BERT key/shape/dtype manifest of the pinned Text2CAD checkpoint.

Reads the official checkpoint with ``torch.load(weights_only=True)`` (structure
only -- no model is built, no forward is run), verifies its SHA256, and writes
the independent manifest that the strict loader and the CPU tests cross-check
against (three-way equality: model <-> checkpoint <-> manifest).

Usage:
    python experiments/scripts/dump_text2cad_expected_keys.py \
        --checkpoint .cache/text2cad_official/text2cad_v1.0/Text2CAD_1.0.pth \
        --out experiments/configs/text2cad_checkpoint_expected_keys_20260930.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import torch

PINNED_CHECKPOINT_SHA256 = "235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598"
OFFICIAL_SOURCE_GIT_SHA = "669a81472ede2269837c22a4d00070eaac360c82"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    actual = sha256(args.checkpoint)
    if actual != PINNED_CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 {actual} != pinned {PINNED_CHECKPOINT_SHA256}")

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or set(blob.keys()) != {"model_state_dict"}:
        raise SystemExit("unexpected checkpoint top-level structure")
    raw = blob["model_state_dict"]
    tensors = {}
    zero_keys = []
    for key, value in sorted(raw.items()):
        if not key.startswith("module."):
            raise SystemExit(f"tensor without module. prefix: {key}")
        stripped = key[len("module."):]
        if stripped.startswith("base_text_embedder."):
            raise SystemExit(f"unexpected BERT tensor in checkpoint: {key}")
        tensors[stripped] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if int(torch.count_nonzero(value).item()) == 0:
            zero_keys.append(stripped)

    manifest = {
        "provenance": {
            "checkpoint_sha256": actual,
            "official_source_git_sha": OFFICIAL_SOURCE_GIT_SHA,
            "generated_by": "experiments/scripts/dump_text2cad_expected_keys.py",
            "generated_on": date.today().isoformat(),
            "structure_only_no_forward": True,
            "note": (
                "zero_tensor_keys are the norm_3 biases of the two decoder layers "
                "without cross-attention (ca_level_start=2): norm_3 is created for "
                "every layer but only applied under use_ca, so those biases kept "
                "their LayerNorm init value 0 (weights kept init value 1)."
            ),
        },
        "n_tensors": len(tensors),
        "n_zero_tensors": len(zero_keys),
        "zero_tensor_keys": zero_keys,
        "tensors": tensors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "n_tensors": len(tensors), "zero_tensor_keys": zero_keys}))


if __name__ == "__main__":
    main()
