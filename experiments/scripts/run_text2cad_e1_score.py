"""Run the admitted Text2CAD E1 matrices on share4 and retain real exits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "src"))

from idea01.text2cad_e1_scoring import (
    STAGES, compare_replay, get_arms, input_path, load_freeze,
    merge_shards, observe_devices, preflight, sha256_file, worker, write_json, write_npz,
)


AUTH_REF = "refs/remotes/origin/codex/e1-fusion360-protocol"
AUTH_PATH = "experiments/configs/idea01_e1_fusion360_run_authorization.json"
MIRROR = "https://github.com/anszliuziang/CADRetrieval"
PAIR_LIMIT = 1_058_304 + 16 * 16 + 2 * 16


def git(*argv: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *argv], text=True).strip()


def authorization(args) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", args.auth_sha):
        raise ValueError("expected a complete authorization commit SHA")
    if git("remote", "get-url", "origin").removesuffix(".git") != MIRROR:
        raise ValueError("execution checkout requires the project mirror")
    if git("rev-parse", AUTH_REF) != args.auth_sha:
        raise ValueError("captured authorization is not on the fetched E1 branch")
    auth = json.loads(git("show", f"{args.auth_sha}:{AUTH_PATH}"))
    if (
        auth["status"] != "pilot_authorized"
        or auth["freeze_sha256"] != sha256_file(args.input / "freeze.json")
        or auth["approved_sha"] != git("rev-parse", "HEAD")
        or git("status", "--porcelain")
    ):
        raise ValueError("source, freeze or approval does not match the prepared run")
    detached = subprocess.run(
        ["git", "-C", str(ROOT), "symbolic-ref", "-q", "HEAD"], capture_output=True,
    )
    if detached.returncode != 1:
        raise ValueError("actual scoring checkout must be detached")
    if (
        type(auth["max_pairs"]) is not int or not 0 < auth["max_pairs"] <= PAIR_LIMIT
        or not math.isfinite(auth["max_card_seconds"]) or not 0 < auth["max_card_seconds"] <= 14400
        or not math.isfinite(auth["deadline_epoch"]) or time.time() >= auth["deadline_epoch"]
        or type(auth["max_devices"]) is not int or not 1 <= auth["max_devices"] <= 2
    ):
        raise ValueError("invalid or expired E1 authorization bounds")
    for path in (args.input.resolve(), args.out.resolve()):
        if path == ROOT or ROOT in path.parents:
            raise ValueError("inputs and results must be outside the experiment source checkout")
    fixed_keys = ROOT / "experiments/configs/text2cad_checkpoint_expected_keys_20260930.json"
    if args.keys_manifest.resolve() != fixed_keys:
        raise ValueError("unrecognized expected checkpoint keys")
    revisions = (
        (args.checkpoint, "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"),
        (args.bert_dir, "6da4b6a26a1877e173fca3225479512db81a5e5b"),
        (args.cad_zip, "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"),
    )
    if any(revision not in str(path.resolve()) for path, revision in revisions):
        raise ValueError("official asset revision paths differ")
    return auth


def stop_owned(processes) -> None:
    for sig, seconds in ((signal.SIGTERM, 7), (signal.SIGKILL, 2)):
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
        until = time.monotonic() + seconds
        while time.monotonic() < until and any(p.poll() is None for p in processes):
            time.sleep(0.1)


def ledger_snapshot(path: Path) -> dict:
    if not path.exists():
        return {"pairs_charged": 0, "card_seconds_spent": 0.0, "invocations": []}
    return json.loads(path.read_bytes())


def pending_time(ledger: dict, now: float) -> float:
    return float(ledger["card_seconds_spent"]) + sum(
        max(0.0, now - row["started_epoch"]) * row["cards_used"]
        for row in ledger["invocations"] if not row.get("finalized")
    )


def wait_for_devices(args, auth, stage, stopping, update) -> list[int]:
    stage_dir = args.out / stage
    update("waiting_for_idle", stage=stage)
    while not stopping() and time.time() < auth["deadline_epoch"] - 15:
        free, smi, mapping = observe_devices()
        (stage_dir / "npu_smi.txt").write_text(smi, encoding="utf-8")
        (stage_dir / "npu_logic_mapping.txt").write_text(mapping, encoding="utf-8")
        if free:
            return free[: 1 if stage == "probe" else auth["max_devices"]]
        until = min(time.time() + 75, auth["deadline_epoch"] - 15)
        while not stopping() and time.time() < until:
            time.sleep(1)
    raise TimeoutError("E1 stopped waiting for share4 devices at the deadline")


def run_workers(args, auth, stage, arms, stopping, update) -> dict:
    stage_dir = args.out / stage
    stage_dir.mkdir()
    devices = wait_for_devices(args, auth, stage, stopping, update)
    tape = args.out / "ledger.json"
    before = ledger_snapshot(tape)
    if any(not r["finalized"] or r["exit_code"] != 0 for r in before["invocations"]):
        raise ValueError("earlier scoring invocations did not all settle successfully")
    pairs = sum(arm.pairs for arm in arms)
    if pairs + before["pairs_charged"] > auth["max_pairs"]:
        raise ValueError("phase would exceed the authorized pair count")
    command = [
        sys.executable, "-B", "-u", __file__, "worker",
        "--input", str(args.input), "--out", str(args.out), "--auth-sha", args.auth_sha,
        "--checkpoint", str(args.checkpoint), "--bert-dir", str(args.bert_dir),
        "--cad-zip", str(args.cad_zip), "--keys-manifest", str(args.keys_manifest),
        "--stage", stage, "--shards", str(len(devices)),
    ]
    commands = [command + ["--device", str(d), "--shard", str(i)] for i, d in enumerate(devices)]
    start = time.time()
    children = []
    write_json(stage_dir / "launch.json", {
        "argv": commands, "selected_logic_ids": devices, "started_epoch": start,
        "planned_pairs": pairs, "auth_sha": args.auth_sha, "freeze_sha256": auth["freeze_sha256"],
        "controller_pid": os.getpid(),
    })
    update("scoring", stage=stage, devices=devices)
    try:
        for i, (argv, device) in enumerate(zip(commands, devices)):
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            with (stage_dir / f"worker{i}.log").open("xb") as output:
                children.append(subprocess.Popen(
                    argv, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                ))
        while any(child.poll() is None for child in children):
            now = time.time()
            elapsed = pending_time(ledger_snapshot(tape), now)
            if (
                stopping() or now >= auth["deadline_epoch"] - 10
                or elapsed + 10 * len(devices) >= auth["max_card_seconds"]
            ):
                raise TimeoutError("E1 scoring interrupted at the wall time or card-second limit")
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("E1 scoring worker failed")
            time.sleep(1)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("E1 scoring worker failed")
    finally:
        stop_owned(children)
        ended = time.time()
        snapshot = ledger_snapshot(tape)
        pending_owned = [
            {
                "invocation_id": entry["invocation_id"], "pid": entry["pid"],
                "pairs_precharged": entry.get("pairs_charged", 0),
                "card_seconds_through_observed_exit": (ended - entry["started_epoch"]) * entry["cards_used"],
            }
            for entry in snapshot["invocations"]
            if not entry["finalized"] and any(
                child.pid == entry["pid"] and child.poll() is not None for child in children
            )
        ]
        write_json(stage_dir / "exits.json", {
            "pids": [child.pid for child in children],
            "return_codes": [child.poll() for child in children],
            "ended_epoch": ended,
            "confirmed_dead_unfinalized_ledger_rows": pending_owned,
            "conservative_card_seconds_at_observation": pending_time(snapshot, ended),
        })
    after = ledger_snapshot(tape)
    measured = after["pairs_charged"] - before["pairs_charged"]
    current_invocations = after["invocations"][len(before["invocations"]):]
    if (
        len(current_invocations) != len(devices)
        or any(not r["finalized"] or r["exit_code"] != 0 for r in current_invocations)
        or measured != pairs or after["card_seconds_spent"] > auth["max_card_seconds"]
    ):
        raise ValueError("actual worker exits, measured pairs or card seconds disagree with the ledger")
    return {
        "stage": stage, "devices": devices, "pairs": measured, "seconds": time.time() - start,
        "cumulative_card_seconds": after["card_seconds_spent"],
    }


def merge_stage(args, arms, stage):
    scores_dir = args.out / "scores"
    for arm in arms:
        paths = sorted((args.out / stage).glob(f"{arm.name}_shard*.npz"))
        matrix, counts = merge_shards(paths, arm.row_ids, arm.column_ids)
        n = len(arm.queries)
        background = matrix[n:]
        if arm.name in ("a", "c_rewrite"):
            main = "b" if arm.name == "a" else "c"
            with np.load(scores_dir / f"{main}.npz", allow_pickle=False) as data:
                if data["cad_uids"].tolist() != arm.column_ids:
                    raise ValueError(f"shared background axes differ for {arm.name}")
                background = data["background"]
        extras = {}
        if arm.name == "c_rewrite":
            extras["query_variants"] = np.array([r["variant"] for r in arm.queries], dtype=str)
        write_npz(
            scores_dir / f"{arm.name}.npz",
            query_uids=np.array([r["uid"] for r in arm.queries], dtype=str),
            cad_uids=np.array(arm.column_ids, dtype=str),
            scores=matrix[:n], background=background, counts=counts, **extras,
        )
    return {arm.name: sha256_file(scores_dir / f"{arm.name}.npz") for arm in arms}


def evaluate(args, source_only, auth, stopping, update):
    out = args.out / ("source_evaluation" if source_only else "evaluation")
    argv = [
        sys.executable, "-B", "-u", str(ROOT / "experiments/scripts/eval_text2cad_e1.py"),
        "--input", str(args.input), "--scores", str(args.out / "scores"), "--output", str(out),
    ]
    if source_only:
        argv.append("--source-only")
    update("evaluating_source" if source_only else "evaluating_final")
    with (args.out / f"{out.name}.log").open("xb") as stream:
        process = subprocess.Popen(
            argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            while process.poll() is None:
                if stopping() or time.time() >= auth["deadline_epoch"] - 10:
                    raise TimeoutError("E1 evaluation reached its execution window")
                time.sleep(1)
        finally:
            stop_owned([process])
            write_json(args.out / f"{out.name}_exit.json", {
                "pid": process.pid, "return_code": process.poll(), "argv": argv,
            })
    if process.returncode != 0:
        raise RuntimeError(f"{out.name} exited {process.returncode}")
    return out


def controller(args, freeze, auth) -> int:
    if os.name != "posix":
        raise ValueError("execution controller needs POSIX process groups")
    if args.out.name != auth["run_id"]:
        raise ValueError("output run_id does not match authorization")
    args.out.mkdir()
    (args.out / "scores").mkdir()
    started = time.time()
    state = {"controller_pid": os.getpid(), "started_epoch": started, "completed_stages": []}
    stop = False

    def update(status, **extra):
        state.update(status=status, updated_epoch=time.time(), **extra)
        write_json(args.out / "state.json", state)

    def request_stop(_sig, _frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    exit_code = 2
    try:
        all_arms = {stage: get_arms(args.input, freeze, stage) for stage in STAGES}
        actual_cap = sum(arm.pairs for arms in all_arms.values() for arm in arms)
        if actual_cap != auth["max_pairs"]:
            raise ValueError("frozen arm pair total differs from the authorized total")
        update("cpu_preflight", freeze_sha256=auth["freeze_sha256"],
               approved_sha=auth["approved_sha"], auth_sha=args.auth_sha,
               authorized_pairs=actual_cap, deadline_epoch=auth["deadline_epoch"])
        counts = {}
        for relpath in freeze["score_inputs_sha256"]:
            input_path(args.input, freeze["score_inputs_sha256"], relpath)
        for arms in all_arms.values():
            for name, values in preflight(args, arms, freeze).items():
                counts[name] = values.tolist()
        write_json(args.out / "preflight.json", {"zero_forward": True, "field_counts": counts})
        for stage, arms in all_arms.items():
            receipt = run_workers(args, auth, stage, arms, lambda: stop, update)
            receipt["score_sha256"] = merge_stage(args, arms, stage)
            write_json(args.out / stage / "matrix_receipt.json", receipt)
            if stage == "probe":
                p = input_path(args.input, freeze["score_inputs_sha256"], "replay_reference.npz")
                with np.load(p, allow_pickle=False) as ref, np.load(
                    args.out / "scores" / "replay.npz", allow_pickle=False
                ) as data:
                    for name in ("query_uids", "cad_uids"):
                        if not np.array_equal(ref[name], data[name]):
                            raise ValueError("historical replay axis differs")
                    replay = compare_replay(data["scores"], data["counts"], ref["scores"], ref["counts"])
                replay.update(reference_sha256=sha256_file(p))
                write_json(args.out / "replay_gate.json", replay)
                if not replay["passed"]:
                    raise ValueError("historical replay tolerance gate failed")
                if "probe" in receipt["score_sha256"]:
                    smoke = json.loads((args.out / "probe" / "worker0_record.json").read_bytes())
                    representatives = smoke["stages"]["probe"]["representatives"]
                    estimate = max(r["seconds"] / r["pairs"] for r in representatives) * (
                        actual_cap - receipt["pairs"]
                    )
                    write_json(args.out / "capacity_estimate.json", {
                        "probes": representatives, "remaining_pairs": actual_cap - receipt["pairs"],
                        "conservative_score_card_seconds_estimate": estimate,
                        "remaining_card_seconds": auth["max_card_seconds"] - receipt["cumulative_card_seconds"],
                        "remaining_wall_seconds": auth["deadline_epoch"] - time.time(),
                    })
            if stage == "source":
                evaluation = evaluate(args, True, auth, lambda: stop, update)
                gate_path = evaluation / "encoding_check.json"
                gate = json.loads(gate_path.read_bytes())
                if gate["status"] not in ("pass", "inconclusive", "fail"):
                    raise ValueError("source encoding assessment has an unknown verdict")
                state["source_encoding"] = gate["status"]
                if gate["status"] == "fail":
                    raise ValueError("source encoding showed clear degradation; target remains sealed")
            state["completed_stages"].append(stage)
            update("stage_complete", stage=stage)
        evaluate(args, False, auth, lambda: stop, update)
        if stop or time.time() >= auth["deadline_epoch"]:
            raise TimeoutError("E1 stopped before final settlement")
        update("completed", pairs=actual_cap)
        exit_code = 0
    except Exception as exc:
        update("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        write_json(args.out / "controller_exit.json", {
            "exit_code": exit_code, "started_epoch": started, "ended_epoch": time.time(),
            "completed_stages": state["completed_stages"],
            "source_encoding": state.get("source_encoding"),
            "ledger": ledger_snapshot(args.out / "ledger.json"),
        })
        entries = [
            f"{sha256_file(path)}  {path.relative_to(args.out).as_posix()}"
            for path in sorted(args.out.rglob("*")) if path.is_file()
        ]
        (args.out / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="ascii", newline="\n")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("controller", "worker", "preflight"))
    for flag in ("input", "out", "checkpoint", "bert-dir", "cad-zip", "keys-manifest"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--auth-sha", required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--device", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int)
    args = parser.parse_args()
    auth = authorization(args)
    freeze = load_freeze(args.input, auth["freeze_sha256"])
    if args.mode == "controller":
        return controller(args, freeze, auth)
    if args.mode == "preflight":
        for stage in STAGES:
            preflight(args, get_arms(args.input, freeze, stage), freeze)
        print(json.dumps({"zero_forward": True, "freeze_sha256": auth["freeze_sha256"]}))
        return 0
    if (
        not args.stage or not args.shards or not 0 <= args.shard < args.shards <= auth["max_devices"]
        or args.device not in (0, 1, 2, 3)
        or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.device)
    ):
        raise ValueError("worker device visibility or shard configuration is invalid")
    require_worker_admission(args, auth)
    worker(args, get_arms(args.input, freeze, args.stage), freeze, auth)
    return 0


def require_worker_admission(args, auth) -> None:
    if args.out.name != auth["run_id"]:
        raise ValueError("worker output run_id differs from authorization")
    launch = json.loads((args.out / args.stage / "launch.json").read_bytes())
    state = json.loads((args.out / "state.json").read_bytes())
    if (
        launch["controller_pid"] != os.getppid()
        or state["controller_pid"] != os.getppid()
        or state["status"] != "scoring" or state["stage"] != args.stage
        or launch["argv"][args.shard][3:] != sys.argv
        or launch["selected_logic_ids"][args.shard] != args.device
        or launch["auth_sha"] != args.auth_sha
        or launch["freeze_sha256"] != auth["freeze_sha256"]
    ):
        raise ValueError("worker is not admitted by the active E1 controller")
    if args.stage != "probe":
        replay = json.loads((args.out / "replay_gate.json").read_bytes())
        if replay["passed"] is not True:
            raise ValueError("historical replay has not passed")
    if args.stage == "target":
        source = json.loads((args.out / "source_evaluation" / "encoding_check.json").read_bytes())
        if source["status"] not in ("pass", "inconclusive"):
            raise ValueError("the source encoding result did not release C")


if __name__ == "__main__":
    raise SystemExit(main())
