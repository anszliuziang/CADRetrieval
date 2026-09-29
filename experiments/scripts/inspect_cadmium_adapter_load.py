"""Read-only, no-forward CADmium adapter load diagnosis for M1b.

Run only from the reviewed Git checkout. This script hashes assets, loads the
base and LoRA on CPU, inspects parameters, and exits without model forward.
"""

from __future__ import annotations

import json
from pathlib import Path

from idea01.cadmium_runner import RunnerConfig, load_model, preflight_assets


def main() -> None:
    root = Path("/home/ma-user/work/CADRetrieval")
    runtime = Path("/home/ma-user/work/CADRetrieval_runtime")
    cfg = RunnerConfig(
        base_model_path=str(runtime / "models/Qwen2.5-Coder-3B-Instruct/488639f1ff808d1d3d0ba301aef8c11461451ec5"),
        adapter_path=str(runtime / "models/CADmium-3B/6c6a79d160595207804171be56a037883b369abd"),
        system_message_config=str(root / "experiments/configs/cadmium_system_message.json"),
        device="cpu",
        asset_manifest=str(root / "experiments/configs/cadmium_asset_sha256_20260930.json"),
    )
    preflight = preflight_assets(cfg)
    model = load_model(cfg)
    lora = [(name, param) for name, param in model.named_parameters() if ".lora_A.default." in name or ".lora_B.default." in name]
    b = [(name, param) for name, param in lora if ".lora_B.default." in name]
    nonzero_b = sum(int(param.count_nonzero().item() > 0) for _, param in b)
    output = {
        "status": "strict_adapter_loaded_no_forward",
        "manifest_checked_files": preflight["manifest_checked_files"],
        "lora_parameter_count": len(lora),
        "lora_B_parameter_count": len(b),
        "nonzero_lora_B_parameter_count": nonzero_b,
        "device": "cpu",
    }
    if len(lora) != 504 or len(b) != 252 or nonzero_b != 252:
        raise RuntimeError(f"CADmium adapter parameter audit failed: {output}")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
