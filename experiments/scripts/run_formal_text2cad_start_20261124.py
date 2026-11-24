"""Admit one source and start its first native Text2CAD epoch on the designated host."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from time import monotonic, sleep


DATASET_REV = "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"
BERT_REV = "6da4b6a26a1877e173fca3225479512db81a5e5b"
NATIVE = "experiments.scripts.run_formal_text2cad_training"
COMPARE = "experiments.scripts.compare_formal_text2cad_updates"
FUSION = "cadmium-fusion-r1.0.1"
SOURCES = {
    "t2cad-v1.0": {"host": "YN-910a-1", "devices": (0, 1), "pairs": 605816, "updates": 37864, "hours": 120,
                   "levels": ("L0", "L1", "L2", "L3")},
    "ct-deepcad": {"host": "YN-910a-2", "devices": (0, 1), "pairs": 148046, "updates": 9253, "hours": 36,
                   "levels": ("CT",)},
    FUSION: {"host": "YN-910a-3", "devices": (1,), "pairs": 2467, "updates": 155, "hours": 6,
             "levels": ("annotation",)},
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def await_exit(path, deadline):
    while not path.is_file() or not path.stat().st_size:
        if monotonic() >= deadline:
            raise TimeoutError(f"Admission step did not settle: {path}")
        sleep(12)
    code = int(path.read_text().strip())
    if code != 0:
        raise RuntimeError(f"Admission step exited {code}: {path}")


def match_recipe(folder, source, ranks, audit, smoke_count, script_sha, devices=None):
    record = read_json(folder / "run.json")
    recipe = record["recipe"]
    if (recipe["source"], recipe["world"], recipe["audit_batch"], recipe["smoke_validation_count"],
            recipe["script_sha256"]) != (source, ranks, audit, smoke_count, script_sha):
        raise ValueError(f"The admission run uses a different source, layout or native implementation: {folder}")
    if devices is not None:
        visible = ",".join(map(str, devices))
        layout = record["layout"]
        if len(layout) != ranks or any(row["visible_devices"] != visible for row in layout):
            raise ValueError(f"The device list differs from the admitted physical layout: {folder}")


def verify_admission(runtime, source, spec):
    qualification = runtime / "provision_20261124_fleet/qualification_00e3d471"
    if not read_json(qualification / "receipt.json")["passed"] or (qualification / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("The host has no passing runtime qualification")
    gates = runtime / "runs/idea01_four_tables_native_training_gate_20261124"
    native_sha = sha256(Path(__file__).with_name("run_formal_text2cad_training.py").read_bytes()).hexdigest()
    deadline = monotonic() + 70 * 60
    await_exit(gates / "tail_single/exit_code.txt", deadline)
    match_recipe(gates / "tail_single/result", source, 1, "tail", 0, native_sha)
    if source == FUSION:
        first = gates / "first_validate"
        await_exit(first / "exit_code.txt", deadline)
        result = first / "result"
    else:
        await_exit(gates / "tail_dual/exit_code.txt", deadline)
        match_recipe(gates / "tail_dual/result", source, 2, "tail", 0, native_sha, spec["devices"])
        recovery = runtime / "runs/idea01_four_tables_native_training_gate_layout_recovery_20261124"
        await_exit(recovery / "exit_code.txt", deadline)
        if not read_json(recovery / "tail_comparison.json")["passed"]:
            raise ValueError("The real single/rank-paired tail batch comparison failed")
        result = recovery / "first_validate"
    match_recipe(result, source, len(spec["devices"]), None, 8, native_sha, spec["devices"])
    summary = read_json(result / "smoke-validation-1.json")["summary"]["levels"]
    if set(summary) != set(spec["levels"]) or any(row["denominator"] != 8 or row["valid"] < 1 for row in summary.values()):
        raise ValueError("The actual fixed validation smoke did not cover its eight-row input")
    return {"tail_single": str(gates / "tail_single/result"), "validation_smoke": str(result), "levels": summary}


def host_health(devices):
    deadline = monotonic() + 45
    while True:
        output = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
        for device in devices:
            if re.search(rf"\|\s+{device}\s+910ProA\s+\|\s+OK\b", output) is None:
                raise RuntimeError(f"Physical NPU {device} is unhealthy")
        if all(f"No running processes found in NPU {device}" in output for device in devices):
            return output
        if monotonic() >= deadline:
            raise RuntimeError(f"Physical NPUs {devices} are busy")
        sleep(5)


def make_native_argv(runtime, source, devices, output, stop, *, resume=False):
    argv = [sys.executable, "-B", "-u", "-m"]
    if len(devices) == 2:
        argv += ["torch.distributed.run", "--standalone", "--nproc_per_node=2", "--max_restarts=0", "-m"]
    argv += [NATIVE, "--source", source, "--device", "npu", "--bert",
             str(runtime / "models/bert-large-uncased" / BERT_REV), "--out", str(output),
             "--stop-after-updates", str(stop), "--save-every", "64"]
    if source == FUSION:
        argv += ["--manifests", str(runtime / "runs/idea01_four_tables_fusion_ridge_20261124/eligibility"),
                 "--cad-zip", str(runtime / "runs/idea01_four_tables_fusion_full_20261124/all/text2cad_candidates.zip"),
                 "--fusion-validation-qrels", str(runtime / "runs/idea01_four_tables_m0_20261124/fusion/validation_qrels.jsonl")]
    else:
        argv += ["--manifests", str(runtime / "four_tables_inputs_20261124/prior_share4/runs/idea01_in_domain_pilot_20261123/manifests"),
                 "--cad-zip", str(runtime / "datasets/Text2CAD-v1.0" / DATASET_REV / "cad_seq.zip")]
    if resume:
        argv.append("--resume")
    return argv


def start_ticks(pid):
    try:
        record = Path(f"/proc/{pid}/stat").read_text()
        return record.rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_children(parent):
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True, timeout=5)
    by_parent = {}
    for line in output.splitlines():
        pid, ppid = map(int, line.split())
        by_parent.setdefault(ppid, []).append(pid)
    result = []
    pending = list(by_parent.get(parent, []))
    while pending:
        pid = pending.pop()
        ticks = start_ticks(pid)
        if ticks is not None:
            result.append((pid, ticks))
        pending.extend(by_parent.get(pid, []))
    return result


def stop_children(children, sig):
    for pid, ticks in children:
        if ticks == start_ticks(pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def ensure_stop(process):
    if process.poll() is not None:
        return process.returncode
    children = owned_children(process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    stop_children(children, signal.SIGTERM)
    try:
        return process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=5)
    finally:
        stop_children(children, signal.SIGKILL)


def execute(control, label, argv, devices, deadline):
    output = control / label
    output.mkdir()
    if devices:
        health = host_health(devices)
        (output / "npu_before.txt").write_text(health, encoding="utf-8")
    remaining = deadline - monotonic() - 20
    if remaining <= 0:
        raise TimeoutError(f"No remaining authorized execution time for {label}")
    environment = os.environ.copy()
    if devices:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    start = monotonic()
    receipt = {"argv": argv, "started_utc": utc(), "visible_devices": devices, "phase_limit_seconds": remaining}
    timed_out = False
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        receipt["pid"] = process.pid
        try:
            save_json(output / "launch.json", receipt)
            actual_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            actual_code = ensure_stop(process)
        except BaseException:
            actual_code = ensure_stop(process)
            save_json(output / "exit.json", {**receipt, "exit_code": actual_code, "interrupted": True, "ended_utc": utc()})
            raise
    save_json(output / "exit.json", {**receipt, "ended_utc": utc(), "wall_seconds": monotonic() - start,
                                     "exit_code": actual_code, "timed_out": timed_out})
    if timed_out or actual_code:
        raise RuntimeError(f"{label} exited {actual_code}; timed_out={timed_out}; evidence at {output}")


def run(args, control, spec):
    runtime = args.runtime
    source = args.source
    devices = spec["devices"]
    admission = verify_admission(runtime, source, spec)
    save_json(control / "admission.json", admission)
    deadline = monotonic() + spec["hours"] * 3600
    formal = control / "formal"
    witness = control / "resume_witness"

    def train(label, dest, steps, *, resume=False, phase_cap=None):
        limit = min(deadline, monotonic() + phase_cap) if phase_cap else deadline
        execute(control, label, make_native_argv(runtime, source, devices, dest, steps, resume=resume), devices, limit)

    train("formal_first_two", formal, 2, phase_cap=3600)
    train("witness_first", witness, 1, phase_cap=3600)
    train("witness_resume", witness, 2, resume=True, phase_cap=3600)
    execute(control, "verify_resume", [sys.executable, "-B", "-m", COMPARE, "resume", str(formal), str(witness)],
            (), min(deadline, monotonic() + 300))
    comparison = read_json(control / "verify_resume/stdout.log")
    if not comparison["passed"] or comparison["updates"] != 2:
        raise ValueError("The first two formal updates cannot be exactly recovered")
    save_json(control / "resume_gate.json", comparison)
    train("formal_first_epoch", formal, spec["updates"], resume=True)

    state = read_json(formal / "checkpoint.json")
    curve = read_json(formal / "curve.json")
    if (state["updates"], state["exposure"], state["epoch"], state["file"]) != (
        spec["updates"], spec["pairs"], 2, "epoch-01.pt",
    ):
        raise ValueError("Completed first-epoch ledger has the wrong exposure or optimizer count")
    if (len(curve) != 1 or curve[0]["epoch"] != 1 or curve[0]["validation"]["nll"] is None
            or set(curve[0]["validation"]["levels"]) != set(spec["levels"])
            or any(row["denominator"] != 512 for row in curve[0]["validation"]["levels"].values())):
        raise ValueError("Expected the fixed full validation panel at the epoch-one checkpoint")
    return {"optimizer_updates": state["updates"], "pairs": state["exposure"], "validation": curve[0]["validation"],
            "checkpoint": state["file"], "weights_sha256": state["weight_sha256"],
            "eligible_epoch": 1, "confirmation_test_scored": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", choices=tuple(SOURCES), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--expected-code", required=True)
    args = parser.parse_args()
    spec = SOURCES[args.source]
    if args.host != spec["host"]:
        raise ValueError("This source is assigned to a different host")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != args.expected_code:
        raise ValueError("The checked-out Git revision does not match authorization")
    if subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode != 1:
        raise ValueError("The executing worktree must be detached")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("The executing worktree is dirty")
    control = args.runtime / "runs/idea01_four_tables_native_formal_20261124" / args.source
    control.mkdir(parents=True, exist_ok=False)
    start = {"source": args.source, "host": args.host, "pid": os.getpid(), "code_sha": args.expected_code,
             "registered_utc": utc(), "devices": spec["devices"], "first_epoch_hours": spec["hours"],
             "max_card_hours_including_witness": spec["hours"] * len(spec["devices"])}
    save_json(control / "registration.json", start)
    print(json.dumps(start), flush=True)

    def interrupt(sig, frame):
        raise InterruptedError(f"Received signal {sig}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        result = run(args, control, spec)
    except BaseException as error:
        save_json(control / "settlement.json", {**start, "status": "blocked", "ended_utc": utc(),
                                                "error_type": type(error).__name__, "detail": str(error)})
        raise
    save_json(control / "settlement.json", {**start, **result, "status": "first_epoch_complete", "ended_utc": utc()})


if __name__ == "__main__":
    main()
