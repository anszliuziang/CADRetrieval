"""No-forward structure audit of pinned Text2CAD weights and one CAD vector.

Reads only official assets in the runtime directory. It neither constructs nor
executes the model; stdout is a compact JSON diagnostic for the Codex review.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import torch


RUNTIME = Path("/home/ma-user/work/CADRetrieval_runtime")
CHECKPOINT = RUNTIME / "models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth"
ARCHIVE = RUNTIME / "datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"
MEMBER = "cad_seq/0000/00000093/seq/00000093.pth"


def describe_tensor(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": str(value.dtype), "nonzero": int(torch.count_nonzero(value).item())}


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise RuntimeError("official checkpoint lacks model_state_dict")
    state = checkpoint["model_state_dict"]
    nonbert = {key: value for key, value in state.items() if not key.startswith("base_text_embedder.")}
    nonzero_nonbert = sum(int(torch.count_nonzero(value).item() > 0) for value in nonbert.values())
    with zipfile.ZipFile(ARCHIVE) as archive:
        sample = torch.load(io.BytesIO(archive.read(MEMBER)), map_location="cpu", weights_only=True)
    vec = sample["vec"]
    masks = sample["mask_cad_dict"]
    cad = vec["cad_vec"]
    pad = masks["key_padding_mask"]
    output = {
        "status": "structure_only_no_forward",
        "checkpoint_top_keys": sorted(checkpoint.keys()),
        "model_state_tensors": len(state),
        "nonbert_tensors": len(nonbert),
        "nonzero_nonbert_tensors": nonzero_nonbert,
        "model_state_first_keys": sorted(state)[:8],
        "model_state_last_keys": sorted(state)[-8:],
        "sample_member": MEMBER,
        "sample_vec_keys": sorted(vec),
        "sample_mask_keys": sorted(masks),
        "cad_vec": describe_tensor(cad),
        "cad_first_rows": cad[:5].tolist(),
        "cad_last_rows": cad[-5:].tolist(),
        "key_padding_mask": describe_tensor(pad),
        "valid_x": int((~pad[:, 0]).sum().item()),
        "valid_y": int((~pad[:, 1]).sum().item()),
    }
    if not nonbert or not nonzero_nonbert:
        raise RuntimeError("checkpoint has no nonzero trained non-BERT parameters")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
