"""Run one approved scoring stage on share4 and retain actual process exits."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git(repo, *argv):
    return subprocess.check_output(["git", "-C", str(repo), *argv], text=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("inputs", "eval", "out", "checkpoint", "bert-dir", "keys-manifest", "cad-zip"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--approved-sha", required=True)
    parser.add_argument("--inputs-sha", required=True)
    parser.add_argument("--eval-sha", required=True)
    parser.add_argument("--text-stop", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--frozen-sha")
    return parser.parse_args()


def stop_owned(children):
    for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        running = [process for process in children if process.poll() is None]
        for process in running:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        until = time.monotonic() + grace
        while any(process.poll() is None for process in running) and time.monotonic() < until:
            time.sleep(0.2)


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    scripts = repo / "experiments" / "scripts"
    if os.name != "posix":
        raise ValueError("stage controller requires Linux process groups")
    if git(repo, "rev-parse", "HEAD") != args.approved_sha or git(repo, "status", "--porcelain"):
        raise ValueError("checkout does not match the approved clean commit")
    if subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"], capture_output=True
    ).returncode != 1:
        raise ValueError("expected a detached checkout")
    if not math.isfinite(args.deadline_epoch) or args.deadline_epoch <= time.time() + 30:
        raise ValueError("no usable time before deadline")
    if len(set(args.devices)) != len(args.devices) or not set(args.devices) <= {0, 1, 2, 3}:
        raise ValueError("expected distinct share4 device indices")
    if sha256(args.inputs) != args.inputs_sha:
        raise ValueError("scoring inputs digest differs")
    if sha256(args.eval) != args.eval_sha:
        raise ValueError("evaluation inputs digest differs")
    inputs = json.loads(args.inputs.read_bytes())
    axis = inputs["axis"]
    if axis == "validation":
        if (
            args.frozen is not None
            or args.frozen_sha is not None
            or args.text_stop != 560
            or len(inputs["members"]) != 8925
        ):
            raise ValueError("validation requires the full preregistered text axis")
    elif axis == "test":
        if args.frozen is None or sha256(args.frozen) != args.frozen_sha:
            raise ValueError("test requires the independently verified frozen configuration")
        if args.text_stop - 48 not in (1024, 768, 512, 256, 128) or len(inputs["members"]) != 8046:
            raise ValueError("test query prefix differs from the preregistered choices")
    else:
        raise ValueError(f"unknown axis {axis!r}")
    if len(inputs["texts"]) < args.text_stop:
        raise ValueError("scoring prefix exceeds the manifest")

    out = args.out.resolve()
    if out == repo or repo in out.parents:
        raise ValueError("run outputs must be outside the source checkout")
    out.mkdir(parents=True)
    scores = out / "scores"
    scores.mkdir()
    commands = [
        [
            sys.executable, "-B", "-u", str(scripts / "run_text2cad_accuracy_night.py"), "score",
            "--inputs", str(args.inputs), "--expected-inputs-sha", args.inputs_sha,
            "--checkpoint", str(args.checkpoint), "--bert-dir", str(args.bert_dir),
            "--keys-manifest", str(args.keys_manifest), "--cad-zip", str(args.cad_zip),
            "--device", f"npu:{device}", "--out", str(scores), "--shard", str(shard),
            "--shards", str(len(args.devices)), "--text-stop", str(args.text_stop),
        ]
        for shard, device in enumerate(args.devices)
    ]
    evaluation = [
        sys.executable, "-B", "-u", str(scripts / "eval_text2cad_accuracy_night.py"),
        "fit" if axis == "validation" else "evaluate",
        "--inputs", str(args.inputs), "--eval", str(args.eval), "--score-dir", str(scores),
        "--out", str(out / "evaluation"),
    ]
    if axis == "test":
        evaluation.extend([
            "--frozen", str(args.frozen), "--expected-frozen-sha", args.frozen_sha,
            "--text-stop", str(args.text_stop),
        ])

    state = {
        "axis": axis,
        "approved_sha": args.approved_sha,
        "controller_pid": os.getpid(),
        "deadline_epoch": args.deadline_epoch,
        "inputs_sha256": args.inputs_sha,
        "eval_sha256": args.eval_sha,
        "frozen_sha256": args.frozen_sha,
        "planned_cads": len(inputs["members"]),
        "planned_texts": args.text_stop,
        "planned_pairs": args.text_stop * len(inputs["members"]),
        "started_epoch": time.time(),
        "worker_argv": commands,
        "evaluation_argv": evaluation,
        "status": "starting",
    }
    write_json(out / "launch.json", state)
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    workers = []
    evaluator = None
    failure = None
    try:
        for shard, command in enumerate(commands):
            with (out / f"worker_{shard}.log").open("xb") as log:
                workers.append(subprocess.Popen(
                    command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                ))
        state.update(status="scoring", worker_pids=[process.pid for process in workers])
        write_json(out / "state.json", state)
        while any(process.poll() is None for process in workers):
            if stopped or time.time() >= args.deadline_epoch - 15:
                raise RuntimeError("requested stop or stage deadline reached")
            if any(process.poll() not in (None, 0) for process in workers):
                raise RuntimeError("score worker failed")
            time.sleep(1)
        if any(process.returncode != 0 for process in workers):
            raise RuntimeError("score worker failed")
        state["scoring_ended_epoch"] = time.time()
        expected = {f"t{i:05d}.npz" for i in range(args.text_stop)}
        actual = {path.name for path in (scores / "chunks").glob("*.npz")}
        if actual != expected or not (scores / "counts.npy").is_file():
            raise RuntimeError("score matrix coverage differs from launch plan")
        if stopped or time.time() >= args.deadline_epoch - 15:
            raise RuntimeError("requested stop or stage deadline reached before evaluation")
        if sha256(args.eval) != args.eval_sha:
            raise RuntimeError("evaluation inputs changed during scoring")
        with (out / "evaluation.log").open("xb") as log:
            evaluator = subprocess.Popen(
                evaluation, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        state.update(status="evaluating", evaluation_pid=evaluator.pid)
        write_json(out / "state.json", state)
        while evaluator.poll() is None:
            if stopped or time.time() >= args.deadline_epoch - 15:
                raise RuntimeError("requested stop or stage deadline reached during evaluation")
            time.sleep(1)
        if evaluator.returncode != 0:
            raise RuntimeError("evaluation failed")
        if stopped or time.time() > args.deadline_epoch:
            raise RuntimeError("requested stop or stage deadline crossed before settlement")
    except (OSError, RuntimeError) as exc:
        failure = str(exc)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_owned(workers + ([evaluator] if evaluator is not None else []))
        state.update(
            error=failure,
            worker_exit_codes=[process.poll() for process in workers],
            evaluation_exit_code=evaluator.poll() if evaluator is not None else None,
            ended_epoch=time.time(),
            status="failed" if failure is not None else "complete",
        )
        write_json(out / "receipt.json", state)
    return 1 if failure is not None else 0


if __name__ == "__main__":
    sys.exit(main())
