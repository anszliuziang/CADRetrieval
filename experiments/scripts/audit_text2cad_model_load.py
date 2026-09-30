"""Zero-forward strict-load audit of the pinned Text2CAD assets (M1c stage B1).

Runs the stage-B1 strict loader against the REAL pinned checkpoint on CPU:
file SHA256 -> checkpoint structure -> key-set/shape equality (model <->
checkpoint <-> committed manifest) -> zero-tensor policy -> per-submodule
``load_state_dict(strict=True)`` -> per-tensor numeric equality. The ported
architecture is built with random init and a stub text embedder; **no forward
is executed anywhere** (the real-model forward budget is 0), and BERT itself
is not loaded here. When ``--bert-dir`` is given, the five pinned BERT files
are SHA-verified; the BERT ``from_pretrained`` loading-info gates run only if
``transformers``/``safetensors`` are importable (they are not part of this
Windows audit box -- the equivalent zero-forward BERT gates already ran on the
server, see experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md,
and re-run on the server at G1 with the reviewed SHA).

stdout is a compact JSON diagnostic for the Codex review; exit code is
non-zero if any gate fails.

Example (local audit box):
    python experiments/scripts/audit_text2cad_model_load.py \
        --checkpoint .cache/text2cad_official/text2cad_v1.0/Text2CAD_1.0.pth \
        --bert-dir .cache/bert_large_uncased \
        --expected-keys experiments/configs/text2cad_checkpoint_expected_keys_20260930.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.text2cad_port.loader import (  # noqa: E402
    AssetVerificationError,
    PINNED_BERT_FILE_SHA256,
    PINNED_CHECKPOINT_SHA256,
    build_text2cad,
    load_verified_checkpoint,
    nonbert_state_dict_keys,
    strict_load_frozen_weights,
    verify_bert_asset_shas,
    verify_bert_model_load,
)
from idea01.text2cad_port.text_embed import TextEmbedder  # noqa: E402


class StubEmbedder(torch.nn.Module):
    """Placeholder so the architecture can be built without transformers."""

    def get_embedding(self, texts):  # pragma: no cover - never called
        raise NotImplementedError("stub embedder must not be called")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bert-dir", type=Path)
    parser.add_argument("--expected-keys", type=Path)
    args = parser.parse_args()

    result = {
        "status": "structure_only_no_forward",
        "checkpoint_path": str(args.checkpoint),
        "pinned_checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
    }
    try:
        state, checkpoint_report = load_verified_checkpoint(args.checkpoint)
        result["checkpoint"] = checkpoint_report

        torch.manual_seed(0)  # random init is overwritten by the strict load
        model = build_text2cad(StubEmbedder(), device="cpu")
        result["model_nonbert_keys"] = len(nonbert_state_dict_keys(model))
        result["strict_load"] = strict_load_frozen_weights(
            model, state, expected_keys_manifest=args.expected_keys
        )
        # Post-load spot facts (structure only, still no forward). eval()
        # switches dropout off without touching weights -- the scoring-time
        # configuration.
        model.eval()
        post = model.state_dict()
        for key in ("cad_decoder.cad_decoder_layers.0.norm_seq.norm_3.weight",
                    "cad_decoder.cad_decoder_layers.1.norm_seq.norm_3.weight"):
            result[f"untouched_init_ones:{key}"] = bool((post[key] == 1).all())
        result["eval_mode"] = not model.training
    except AssetVerificationError as exc:
        result["status"] = "asset_verification_failed"
        result["reason"] = exc.reason
        result["detail"] = str(exc)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1) from exc

    if args.bert_dir is not None:
        result["bert_files"] = {
            name: info["sha256"] for name, info in verify_bert_asset_shas(args.bert_dir).items()
        }
        has_transformers = importlib.util.find_spec("transformers") is not None
        has_safetensors = importlib.util.find_spec("safetensors") is not None
        if has_transformers and has_safetensors:
            embedder = TextEmbedder.from_local(args.bert_dir, max_seq_len=512, device="cpu")
            result["bert_load"] = verify_bert_model_load(embedder, args.bert_dir)
        else:
            result["bert_load"] = {
                "skipped": "transformers/safetensors not importable on this host",
                "prior_zero_forward_evidence": (
                    "experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md: "
                    "391/391 tensors, missing/mismatched empty, 7 cls.* unused, "
                    "word embeddings exactly equal (server audit, 2026-09-30)"
                ),
                "rerun_required": "execute this script on yunnan-share2 at G1 with the reviewed SHA",
            }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
