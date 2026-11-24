"""Compare real optimizer witnesses or uninterrupted/resumed runs."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def equal_tree(a, b, name):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b) or a.dtype != b.dtype:
            raise ValueError(f"Resume tensor differs: {name}")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b) or a.dtype != b.dtype:
            raise ValueError(f"Resume array differs: {name}")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError(f"Resume keys differ: {name}")
        for k in a:
            equal_tree(a[k], b[k], f"{name}.{k}")
    elif isinstance(a, (list, tuple)):
        for i, (left, right) in enumerate(zip(a, b, strict=True)):
            equal_tree(left, right, f"{name}.{i}")
    elif a != b:
        raise ValueError(f"Resume scalar differs: {name}")


def checkpoint(root):
    pointer = json.loads((root / "checkpoint.json").read_bytes())
    return load(root / pointer["file"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("ddp", "resume", "padding"))
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--loss-atol", type=float, default=1e-5)
    p.add_argument("--gradient-atol", type=float, default=1e-6)
    p.add_argument("--update-atol", type=float, default=2e-6)
    args = p.parse_args()
    metrics = dict(kind=args.kind)
    if args.kind == "resume":
        a, b = checkpoint(args.reference), checkpoint(args.candidate)
        equal_tree(a, b, "checkpoint")
        metrics.update(updates=a["state"]["updates"], exposure=a["state"]["exposure"],
                       weight_sha256=a["weight_sha256"])
    else:
        a, b = (load(root / "update-witness.pt") for root in (args.reference, args.candidate))
        recipes = [json.loads((root / "run.json").read_bytes())["recipe"] for root in (args.reference, args.candidate)]
        expected_world = (1, 2)
        if args.kind == "padding":
            expected_world = (2, 2)
            multiple = recipes[1].pop("training_text_pad_multiple", None)
            if "training_text_pad_multiple" in recipes[0] or multiple not in (32, 64):
                raise ValueError("Padding witness needs a natural reference and a fixed-width candidate")
            metrics["padding_multiple"] = multiple
            equal_tree(a["by_rank"], b["by_rank"], "rank sample identities and text lengths")
            widths = [[rank["text_width"] for rank in row["timing_by_rank"]] for row in (a, b)]
            if widths[1] != [(width + multiple - 1) // multiple * multiple for width in widths[0]]:
                raise ValueError("Observed padded widths differ from registered layout")
            metrics["text_widths"] = widths
        if tuple(recipe.pop("world") for recipe in recipes) != expected_world or recipes[0] != recipes[1]:
            raise ValueError("Witness recipes differ outside the registered layout")
        for name in ("epoch", "batch", "update", "pairs", "fields", "pair_ids", "truncated", "by_level", "lr"):
            equal_tree(a[name], b[name], name)
        shards = b["by_rank"]
        if (len(shards) != 2 or len(set(a["pair_ids"])) != a["pairs"]
                or [row["pair_ids"] for row in shards] != [a["pair_ids"][rank::2] for rank in range(2)]
                or [sum(row["fields"][head] for row in shards) for head in (0, 1)] != a["fields"]):
            raise ValueError("Distributed shards do not cover the same rows and effective fields")
        if len({width for row in shards for width in row["text_lengths"]}) < 2 or not 0 < a["fields"][1] < a["fields"][0]:
            raise ValueError("The witness needs different text lengths and missing y fields")
        for name in ("loss", "grad_norm"):
            if not math.isclose(a[name], b[name], rel_tol=args.rtol, abs_tol=args.loss_atol):
                raise ValueError(f"{name} differs: {a[name]} vs {b[name]}")
        if args.kind == "padding":
            for name in ("smoothed_sum", "nll_sum"):
                for head, (x, y) in enumerate(zip(a[name], b[name], strict=True)):
                    fields = a["fields"][head]
                    if not math.isclose(x / fields, y / fields, rel_tol=args.rtol, abs_tol=args.loss_atol):
                        raise ValueError(f"{name} head {head} differs after normalization")
        metrics.update(pair_ids=a["pair_ids"], loss_abs_error=abs(a["loss"]-b["loss"]),
                       fields=a["fields"], distributed_shards=shards, probes={},
                       tolerances=dict(rtol=args.rtol, loss_atol=args.loss_atol,
                                       gradient_atol=args.gradient_atol, update_atol=args.update_atol))
        for name, ref in a["parameters"].items():
            other = b["parameters"][name]
            equal_tree(ref["before"], other["before"], name)
            delta = ref["after"] - ref["before"]
            other_delta = other["after"] - other["before"]
            if not delta.abs().max().item() or not ref["gradient"].abs().max().item():
                raise ValueError(f"Missing real update/gradient in {name}")
            torch.testing.assert_close(other["gradient"], ref["gradient"], atol=args.gradient_atol, rtol=args.rtol)
            torch.testing.assert_close(other_delta, delta, atol=args.update_atol, rtol=args.rtol)
            metrics["probes"][name] = dict(gradient_max_abs_error=(ref["gradient"]-other["gradient"]).abs().max().item(),
                                          update_max_abs_error=(delta-other_delta).abs().max().item())
    print(json.dumps(dict(passed=True, **metrics), allow_nan=False))


if __name__ == "__main__":
    main()
