"""Bounded numerical admission checks for a formal SpectraBRep first update."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def parameter_probes(towers):
    """Track a zero-initialized LoRA B matrix and the CAD output projection."""
    adapter = next((p for name, p in towers.text.backbone.named_parameters()
                    if ".lora_B." in name and p.requires_grad), None)
    if adapter is None:
        raise RuntimeError("The official text backbone has no trainable LoRA B matrix")
    selected = {"lora_B": adapter, "cad_output_weight": towers.cad.output[-1].weight}
    return {name: (p, p.detach().cpu().clone()) for name, p in selected.items()}


def update_evidence(probes):
    changes = {}
    for name, (parameter, before) in probes.items():
        if parameter.grad is None:
            raise RuntimeError(f"The first update produced no gradient for {name}")
        gradient = parameter.grad.detach().cpu().abs().max().item()
        delta = (parameter.detach().cpu() - before).abs().max().item()
        if not (0 < gradient < float("inf") and 0 < delta < float("inf")):
            raise RuntimeError(f"No finite nonzero gradient and update for {name}: grad={gradient}, delta={delta}")
        changes[name] = {"clipped_gradient_abs_max": gradient, "parameter_change_abs_max": delta}
    return changes


@torch.no_grad()
def checkpoint_replay(args, towers, upstream, tokenizer, mean, std, config, inputs, cad):
    """Reload an independent base, adapter, both heads and normalization on the same training batch."""
    towers.eval()
    reference = towers(inputs, cad)
    runtime = upstream["runtime"]
    with TemporaryDirectory(prefix=".witness-", dir=args.output) as temporary:
        root = Path(temporary)
        runtime.save_checkpoint(root, towers.cad, towers.text, tokenizer, mean, std, config)
        checkpoint = runtime.load_checkpoint(root / "retrieval.pt")
        if not (torch.equal(checkpoint["mean"], mean) and torch.equal(checkpoint["std"], std)):
            raise ValueError("Restored source normalization differs from the full training-axis fit")
        restored_text = upstream["model"].TextEncoder(
            str(args.qwen_dir), lora_rank=16, lora_alpha=32, adapter=str(root / "adapter"), dtype=torch.float32,
        )
        runtime.restore_text_heads(restored_text, checkpoint)
        restored_cad = upstream["model"].CADTower()
        restored_cad.load_state_dict(checkpoint["cad_tower"], strict=True)
        restored = type(towers)(restored_text, restored_cad).to(cad.device).eval()
        recovered = restored(inputs, cad)
        errors = {}
        for axis, expected, actual in zip(("text", "cad"), reference, recovered, strict=True):
            expected, actual = expected.cpu(), actual.cpu()
            if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                raise ValueError(f"Nonfinite {axis} embedding during checkpoint replay")
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            errors[axis] = float((expected - actual).abs().max())
    return {
        "status": "passed", "batch_pairs": len(cad), "embedding_max_absolute_difference": errors,
        "atol": 1e-5, "rtol": 1e-5,
        "normalization_sha256": {name: sha256(tensor.numpy().tobytes()).hexdigest()
                                 for name, tensor in (("mean", mean), ("std", std))},
    }
