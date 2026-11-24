"""Resume the original trainer only after verifying the restored state before any update."""

import argparse
import json
from pathlib import Path
from struct import pack
import sys


def byte_tree(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return str(tensor.dtype), tuple(tensor.shape), tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    if isinstance(value, np.ndarray):
        return str(value.dtype), value.shape, value.tobytes()
    if isinstance(value, float):
        return pack(">d", value)
    if isinstance(value, dict):
        return {key: byte_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(byte_tree(item) for item in value)
    return value


def main():
    # Run the trainer from its original immutable checkout. Its recipe validates
    # both the exact file and Git revision against the incoming checkpoint.
    sys.path.insert(0, str(Path.cwd()))
    import torch

    from experiments.scripts import run_formal_text2cad_training as training
    from experiments.scripts.compare_formal_text2cad_updates import equal_tree

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--restore-file-sha256", required=True)
    verification, trainer_args = parser.parse_known_args()
    if "--resume" not in trainer_args:
        raise ValueError("This entry point requires an existing checkpoint and --resume")
    sys.argv[1:] = trainer_args
    restore = training.restore

    def restore_and_verify(args, recipe, model, optimizer, device, rank):
        pointer_path = args.out / "checkpoint.json"
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes)
        path = args.out / pointer["file"]
        if path.parent != args.out or training.sha256_file(path) != verification.restore_file_sha256:
            raise ValueError("Incoming checkpoint bytes differ from the approved SHA-256")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = restore(args, recipe, model, optimizer, device, rank)
        if pointer_path.read_bytes() != pointer_bytes:
            raise ValueError("Checkpoint changed during verified restoration")
        weights = training.cpu_weights(model)
        equal_tree(checkpoint["weights"], weights, "restored.weights")
        if training.weight_sha(weights) != pointer["weight_sha256"]:
            raise ValueError("Restored parameter bytes differ from the checkpoint")
        for parameter, moments in optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq"):
                if moments[name].device != parameter.device:
                    raise ValueError(f"Restored {name} is on a different device from its parameter")
        equal_tree(byte_tree(checkpoint["optimizer"]), byte_tree(optimizer.state_dict()), "restored.optimizer")
        equal_tree(byte_tree(checkpoint["rng_by_rank"][rank]), byte_tree(training.rng(device)), "restored.rng")
        equal_tree(checkpoint["state"], state, "restored.state")
        training.write_json(args.out / f"exact_restore_step_{state['updates']}_rank_{rank}.json", {
            "passed": True, "rank": rank, "updates": state["updates"], "exposure": state["exposure"],
            "epoch": state["epoch"], "next_batch": state["next_batch"], "checkpoint_file": pointer["file"],
            "weight_sha256": pointer["weight_sha256"], "restore_file_sha256": verification.restore_file_sha256,
            "source_code_sha256": recipe["script_sha256"],
            "source_git_sha": recipe["code_sha"], "verified": ["weights", "optimizer", "rng", "training_state"],
        })
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=training.CONTROL_GROUP)
        return state

    training.restore = restore_and_verify
    training.main()


if __name__ == "__main__":
    main()
