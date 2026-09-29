"""Read-only SHA and nonzero LoRA tensor audit of a safetensors checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    h = hashlib.sha256()
    with args.path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    with safe_open(str(args.path), framework="np", device="cpu") as sf:
        keys = list(sf.keys())
        a = [key for key in keys if key.endswith(".lora_A.weight")]
        b = [key for key in keys if key.endswith(".lora_B.weight")]
        nonzero_b = [key for key in b if np.count_nonzero(sf.get_tensor(key))]
    print(json.dumps({
        "path": str(args.path.resolve()),
        "sha256": h.hexdigest(),
        "total_tensors": len(keys),
        "lora_A_tensors": len(a),
        "lora_B_tensors": len(b),
        "nonzero_lora_B_tensors": len(nonzero_b),
        "lora_delta_identically_zero": len(b) > 0 and not nonzero_b,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
