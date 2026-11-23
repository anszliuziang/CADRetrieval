#!/usr/bin/env python3
"""Small, isolated runtime witnesses for the 910A machine's project environment."""

import argparse
from datetime import timedelta
from importlib.metadata import version
import json
import os
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def report(**values):
    print(json.dumps(values, allow_nan=False, sort_keys=True), flush=True)


def seeded(shape):
    return torch.randn(shape, generator=torch.Generator().manual_seed(17))


def check_tensor(name, value, reference, tolerance):
    if value.device.type != "npu":
        raise RuntimeError(f"{name}: expected an NPU output")
    actual = value.detach().float().cpu()
    delta = (actual - reference).abs()
    report(test=name, shape=list(actual.shape), finite=bool(actual.isfinite().all()),
           max_abs_delta=float(delta.max()), tolerance=tolerance)
    torch.testing.assert_close(actual, reference, rtol=0, atol=tolerance)


def fp32_witness(device):
    query = seeded((1, 4, 24, 32))
    key = seeded((1, 4, 24, 32)).flip(-1).contiguous()
    value = seeded((1, 4, 24, 32)).flip(-2).contiguous()
    mask = torch.triu(torch.full((24, 24), float("-inf")), diagonal=1)
    q, k, v, m = (tensor.to(device) for tensor in (query, key, value, mask))
    deltas = (q @ k.transpose(-2, -1)) / (32 ** 0.5) + m
    reference = ((query @ key.transpose(-2, -1)) / (32 ** 0.5) + mask).softmax(-1) @ value
    check_tensor("fp32_masked_attention", deltas.softmax(-1) @ v, reference, 1e-4)
    labels = torch.arange(24, dtype=torch.long) % 16
    logits = seeded((24, 16))
    check_tensor("fp32_smoothed_cross_entropy",
                 F.cross_entropy(logits.to(device), labels.to(device), label_smoothing=0.1),
                 F.cross_entropy(logits, labels, label_smoothing=0.1), 1e-4)
    torch.npu.synchronize(device)


def fp16_witness(device):
    x = seeded((16, 32)).half()
    weights = seeded((20, 32)).half()
    model = torch.nn.Linear(32, 20, bias=False, device=device, dtype=torch.float16)
    with torch.no_grad():
        model.weight.copy_(weights.to(device))
    output = model(x.to(device))
    check_tensor("fp16_linear", output, x.float() @ weights.float().T, 2e-2)
    labels = torch.arange(16, dtype=torch.long, device=device)
    loss = F.cross_entropy(output.float(), labels)
    loss.backward()
    if not bool(model.weight.grad.isfinite().all()):
        raise ValueError("non-finite FP16 gradient")
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    optimizer.step()
    torch.npu.synchronize(device)
    changed = not torch.equal(before, model.weight)
    if not changed or not bool(model.weight.isfinite().all()):
        raise ValueError("the FP16 optimizer witness did not update a finite parameter")
    report(test="fp16_parameter_update", loss=float(loss.detach()), parameter_changed=changed)


def distributed_witness():
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=75))
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        value = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(value)
        torch.npu.synchronize(device)
        expected = world * (world + 1) / 2
        if float(value) != expected:
            raise ValueError(f"incorrect all-reduce on rank {rank}")
        report(test="hccl_all_reduce", rank=rank, world_size=world, actual=float(value), expected=expected)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fp32", "fp16", "bf16-capability", "hccl"))
    args = parser.parse_args()
    if args.mode == "hccl":
        distributed_witness()
        return
    torch.set_num_threads(2)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.init()
    facts = {"mode": args.mode, "name": torch.npu.get_device_name(0),
             "torch": version("torch"), "torch_npu": version("torch-npu"),
             "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}
    if args.mode == "bf16-capability":
        query = getattr(torch.npu, "is_bf16_supported", None)
        report(**facts, bf16_supported=bool(query()) if query is not None else None)
        return
    started = time.perf_counter()
    if args.mode == "fp32":
        fp32_witness(device)
    else:
        fp16_witness(device)
    report(**facts, witness_passed=True, wall_seconds=time.perf_counter() - started)


if __name__ == "__main__":
    main()
