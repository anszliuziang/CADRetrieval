"""Continue an admitted Text2CAD run after its complete first epoch settles.

Invoke from the new detached server checkout with its absolute script path. The
original checkout and artifacts are read-only. A single continued process uses
the original per-epoch wall-clock cap, measured from launch or the previous
committed epoch, including any candidate validation. The trainer owns selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import file_digest, sha256
import json
import math
import os
from pathlib import Path
import re
from shutil import copyfile
import signal
import socket
import subprocess
import time

if __package__:
    from . import run_formal_text2cad_start_20261124 as start
else:
    import run_formal_text2cad_start_20261124 as start


FIRST_RUNS = Path("runs/idea01_four_tables_native_formal_20261124")
SUCCESSORS = Path("runs/idea01_four_tables_native_successors_20261124")
TRAINER = Path("experiments/scripts/run_formal_text2cad_training.py")
RESTORE = Path("experiments/scripts/run_formal_text2cad_verify_restore.py")
RESTORE_SHA = "b15822d20b03abc314da9b7285ced609159d227cd03c540c1a8958327c76a84c"
T_TRAINER = "896f7160804adca5c7fed474d1f37904b7789a1b"
CT_F_TRAINER = "a04640d12f05d95851a4edade838fb84a1d39685"
HCCL_BASE_PORT = 21456
SETTLEMENT_SECONDS = 120
POLL_SECONDS = 12
STOP_RESERVE_SECONDS = 20
GLOBAL_BATCH = 16


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def git(worktree, *arguments):
    return subprocess.check_output(["git", "-C", str(worktree), *arguments], text=True, timeout=30).strip()


def verify_checkout(worktree, expected=None):
    revision = git(worktree, "rev-parse", "HEAD")
    require(expected is None or revision == expected, f"Unexpected checkout revision at {worktree}")
    require(Path(git(worktree, "rev-parse", "--show-toplevel")).resolve() == worktree,
            f"Expected a checkout root at {worktree}")
    symbolic = subprocess.run(["git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"], timeout=30)
    require(symbolic.returncode == 1, f"Expected a detached checkout at {worktree}")
    require(not git(worktree, "status", "--porcelain", "--untracked-files=all"), f"Dirty checkout at {worktree}")
    return revision


def options(tokens):
    remaining = [token for token in tokens if token != "--resume"]
    require(len(tokens) == len(remaining) + 1 and not len(remaining) % 2, "Invalid first-epoch trainer options")
    pairs = dict(zip(remaining[::2], remaining[1::2], strict=True))
    require(len(pairs) == len(remaining) // 2, "Duplicated first-epoch trainer option")
    return pairs


def previous_request(runtime, source):
    spec = start.SOURCES[source]
    suffix = "_exact_restore_v2" if source == start.FUSION else "_exact_restore"
    previous = runtime / FIRST_RUNS / f"{source}{suffix}"
    formal = previous / "formal"
    launch = start.read_json(previous / "launch.json")
    revision = T_TRAINER if source == "t2cad-v1.0" else CT_F_TRAINER
    devices = spec["devices"]
    visible = launch.get("physical_npus", [launch.get("physical_npu")])
    require(launch["host"] == spec["host"] and tuple(visible) == devices and launch["trainer_git_sha"] == revision,
            "First-epoch host, devices, or original trainer revision differs from the register")
    require(type(launch["deadline_epoch"]) is int and launch["deadline_epoch"] > 0 and
            launch["starting_updates"] == 2 and is_sha(launch["checkpoint_file_sha256"]),
            "First-epoch deadline or incoming checkpoint pin is missing")
    if len(devices) == 2:
        old_port = launch["hccl_if_base_port"]
        require(type(old_port) is int and 1024 <= old_port <= 65520 and abs(old_port - HCCL_BASE_PORT) >= 16,
                "Continuation and first-epoch HCCL port blocks overlap or are invalid")
    argv = start.read_json(previous / "actual_argv.json")
    require(isinstance(argv, list) and all(isinstance(token, str) for token in argv) and argv and
            Path(argv[0]).is_absolute(), "First-epoch actual argv has no absolute interpreter")
    prefix = [argv[0], "-B", "-u"]
    if len(devices) == 2:
        prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "--max_restarts=0"]
    offset = len(prefix)
    require(argv[:offset] == prefix and len(argv) > offset + 3 and
            Path(argv[offset]).is_absolute() and Path(argv[offset]).name == RESTORE.name and
            argv[offset + 1:offset + 3] == ["--restore-file-sha256", launch["checkpoint_file_sha256"]],
            "First-epoch actual argv differs from the recorded verified launch")
    original_options = options(argv[offset + 3:])
    expected = start.make_native_argv(runtime, source, devices, formal, spec["updates"], resume=True)
    require(original_options == options(expected[expected.index(start.NATIVE) + 1:]),
            "First-epoch actual trainer options differ from the registered inputs or update bound")
    worktree = (runtime / "worktrees" / f"cad-fleet-{revision[:7]}").resolve(strict=True)
    return previous, launch, worktree, prefix, original_options


def await_first_exit(path, deadline, *, wait=True):
    settle_by = deadline + SETTLEMENT_SECONDS
    while True:
        if path.is_file():
            result = path.read_text(encoding="utf-8").strip()
            if result:
                if path.stat().st_mtime > settle_by:
                    raise TimeoutError(f"First-epoch exit exceeded its settlement window: {path}")
                return int(result)
        remaining = settle_by - time.time()
        if remaining <= 0:
            raise TimeoutError(f"First-epoch actual exit missing after the original deadline: {path}")
        if not wait:
            return None
        time.sleep(min(POLL_SECONDS, remaining))


def verify_layout(record, spec, revision, script_sha, source, milestone):
    recipe = record["recipe"]
    ranks = len(spec["devices"])
    expected = dict(source=source, world=ranks, device="npu", seed=17, global_batch=GLOBAL_BATCH,
                    train_pairs=spec["pairs"], updates_per_epoch=spec["updates"], levels=list(spec["levels"]),
                    base_candidates=[epoch for epoch in (1, 2, 4, 8, 16) if epoch <= milestone],
                    conditional_extension=2 * milestone, script_sha256=script_sha, code_sha=revision,
                    train_only=False, audit_batch=None, smoke_validation_count=0)
    require(all(recipe[key] == value for key, value in expected.items()), "Native trainer recipe or input axes differ")
    layout = sorted(record["layout"], key=lambda row: row["rank"])
    visible = ",".join(map(str, spec["devices"]))
    require(len(layout) == ranks and all(row["rank"] == rank and row["device"] == f"npu:{rank}" and
            row["visible_devices"] == visible and row["pid"] > 0 for rank, row in enumerate(layout)) and
            len({row["pid"] for row in layout}) == ranks and len({row["hostname"] for row in layout}) == 1 and
            len({row.get("visible_ascend_devices") for row in layout}) == 1, "Native rank layout differs")
    return layout


def verify_inputs(arguments, recipe):
    manifests = Path(arguments["--manifests"])
    fusion = "--fusion-validation-qrels" in arguments
    receipt = manifests / ("eligibility_manifest.json" if fusion else "manifest_receipt.json")
    require(digest(receipt) == recipe["formal_receipt"] and is_sha(recipe["cad_zip_sha256"]) and
            Path(arguments["--cad-zip"]).is_file() and Path(arguments["--bert"]).is_dir(),
            "First-epoch input receipts or pinned paths differ")
    if fusion:
        qrels = Path(arguments["--fusion-validation-qrels"])
        require(digest(qrels) == start.read_json(receipt)["validation_qrels_sha256"], "Fusion validation qrels differ")


def verify_restore_receipts(formal, spec, revision, script_sha, checkpoint, byte_sha):
    weights = []
    for rank in range(len(spec["devices"])):
        updates = checkpoint["updates"]
        witness = start.read_json(formal / f"exact_restore_step_{updates}_rank_{rank}.json")
        expected = dict(passed=True, rank=rank, updates=updates, exposure=checkpoint["exposure"],
                        epoch=checkpoint["epoch"], next_batch=checkpoint["next_batch"], source_git_sha=revision,
                        source_code_sha256=script_sha, restore_file_sha256=byte_sha)
        if "file" in checkpoint:
            expected.update(checkpoint_file=checkpoint["file"], weight_sha256=checkpoint["weight_sha256"])
        require(all(witness[key] == value for key, value in expected.items()) and
                set(witness["verified"]) == {"weights", "optimizer", "rng", "training_state"} and
                is_sha(witness["weight_sha256"]), f"Exact-restore evidence differs for rank {rank}")
        weights.append(witness["weight_sha256"])
    require(len(set(weights)) == 1, "Exact-restore weight SHA differs across ranks")


def verify_steps(path, spec, epochs):
    batches = spec["updates"]
    pairs_per_epoch = spec["pairs"]
    with path.open(encoding="utf-8") as stream:
        for update in range(1, epochs * batches + 1):
            epoch, batch = divmod(update - 1, batches)
            prior = batch * GLOBAL_BATCH
            pairs = min(GLOBAL_BATCH, pairs_per_epoch - prior)
            line = stream.readline()
            expected = dict(epoch=epoch + 1, batch=batch, update=update, pairs=pairs,
                            exposure=epoch * pairs_per_epoch + prior + pairs)
            require(line.endswith("\n"), f"Step ledger is incomplete at update {update}")
            row = json.loads(line)
            require(all(row[key] == value for key, value in expected.items()), f"Step ledger differs at update {update}")
        require(not stream.read(1), "Step ledger contains updates beyond the committed epoch")


def verify_validation(formal, epoch, validation, support, levels):
    evidence = start.read_json(formal / f"validation-{epoch:02}.json")
    rows = evidence["rows"]
    require(evidence["summary"] == validation and len(rows) == len(support) == 512 * len(levels) and
            set(validation["levels"]) == set(levels), f"Epoch {epoch} fixed validation evidence differs")
    require(all(all(row[key] == value for key, value in original.items())
                for row, original in zip(rows, support, strict=True)), f"Epoch {epoch} validation support changed")
    for level in levels:
        level_rows = [row for row in rows if row["level"] == level]
        good = [row for row in level_rows if row["status"] == "ok"]
        fields = sum(row["fields"] for row in good)
        nll = math.fsum(row["nll"] for row in good)
        uid_bytes = "".join(f"{row['uid']}\n" for row in good).encode("utf-8")
        require([row["index"] for row in level_rows] == list(range(512)) and
                len({row["uid"] for row in level_rows}) == 512 and fields > 0 and math.isfinite(nll),
                f"Epoch {epoch} validation input axis differs for {level}")
        expected = dict(denominator=512, valid=len(good), statuses=dict(Counter(row["status"] for row in level_rows)),
                        fields=fields, total_nll=nll, nll=nll / fields, valid_uid_sha256=sha256(uid_bytes).hexdigest())
        require(validation["levels"][level] == expected, f"Epoch {epoch} validation summary differs for {level}")
    nll = math.fsum(validation["levels"][level]["nll"] for level in levels) / len(levels)
    require(validation["nll"] == nll, f"Epoch {epoch} pooled native validation loss differs")


def verify_formal(formal, spec, milestone, *, completed):
    checkpoint = start.read_json(formal / "checkpoint.json")
    curve = start.read_json(formal / "curve.json")
    epochs = len(curve)
    require((milestone <= epochs <= 2 * milestone if completed else epochs == 1) and
            checkpoint["history"] == curve and checkpoint["completed"] is completed and
            checkpoint["epoch"] == epochs + 1 and checkpoint["next_batch"] == 0 and
            checkpoint["updates"] == epochs * spec["updates"] and checkpoint["exposure"] == epochs * spec["pairs"] and
            checkpoint["stop_epoch"] == (epochs if completed else milestone) and
            checkpoint["file"] == f"epoch-{epochs:02}.pt" and is_sha(checkpoint["weight_sha256"]),
            "Checkpoint does not record a fully committed epoch and the expected training exposure")
    support = start.read_json(formal / "validation-support.json")
    levels = spec["levels"]
    for epoch, row in enumerate(curve, start=1):
        require(row["epoch"] == epoch and row["global_update"] == epoch * spec["updates"] and
                row["train"]["exposure"] == spec["pairs"] and row["train"]["by_level"] ==
                dict.fromkeys(levels, spec["pairs"] // len(levels)) and (formal / f"epoch-{epoch:02}.pt").is_file(),
                f"Epoch {epoch} curve, checkpoint, or exposure ledger differs")
        candidate = epoch in (1, 2, 4, 8, 16, 32) and (epoch <= milestone or epoch == 2 * milestone)
        require(("validation" in row) == candidate, f"Epoch {epoch} is outside the registered validation schedule")
        if candidate:
            verify_validation(formal, epoch, row["validation"], support, levels)
    verify_steps(formal / "steps.jsonl", spec, epochs)
    return checkpoint, curve


def check_selection(curve, selected, spec, milestone):
    candidates = {row["epoch"]: row["validation"]["nll"] for row in curve if "validation" in row}
    extended = candidates[milestone] < min(nll for epoch, nll in candidates.items() if epoch < milestone)
    terminal = 2 * milestone if extended else milestone
    require(len(curve) == terminal, "Native selection ended at an unexpected terminal epoch")
    best = min(candidates, key=lambda epoch: (candidates[epoch], epoch))
    trained_pairs = sum(row["train"]["exposure"] for row in curve)
    expected = dict(epoch=best, file=f"epoch-{best:02}.pt", nll=candidates[best], exposure=trained_pairs,
                    candidates=list(candidates), improved_at_budget=best == terminal)
    require(selected == expected, "Native selection artifact differs from the registered earliest minimum")
    return dict(selected_epoch=best, selected_validation_nll=candidates[best], selected_checkpoint=selected["file"],
                selected_updates=best * spec["updates"], selected_pairs=sum(row["train"]["exposure"] for row in curve[:best]),
                final_epoch=terminal, final_updates=terminal * spec["updates"], final_pairs=trained_pairs,
                improved_at_budget=selected["improved_at_budget"], extended=extended, candidates=selected["candidates"])


def clone_epoch(previous, formal, byte_sha):
    formal.mkdir()
    names = ("checkpoint.json", "epoch-01.pt", "curve.json", "steps.jsonl", "validation-support.json", "validation-01.json")
    for name in (*names, "run.json"):
        incoming = previous / name
        if name == "run.json" and not incoming.is_file():
            continue
        outgoing = formal / name
        copyfile(incoming, outgoing)
        expected = byte_sha if name == "epoch-01.pt" else digest(incoming)
        require(digest(outgoing) == expected, f"First-epoch snapshot differs: {incoming}")


def watch_training(process, formal, cap, started, wall_started):
    committed = 1
    deadline = started + cap
    while True:
        code = process.poll()
        if code is not None and code != 0:
            return code
        with (formal / "checkpoint.json").open(encoding="utf-8") as stream:
            pointer = json.load(stream)
            pointer_time = os.fstat(stream.fileno()).st_mtime
        finished = pointer["epoch"] - 1
        require(committed <= finished <= 32, "Native checkpoint epoch cursor moved out of bounds")
        for epoch in range(committed + 1, finished + 1):
            epoch_file = f"epoch-{epoch:02}.pt"
            # The latest pointer is the commit timestamp when it still points to
            # the epoch file. An older file time gives a conservative bound if
            # a later progress checkpoint has already replaced that pointer.
            committed_at = pointer_time if pointer["file"] == epoch_file else (formal / epoch_file).stat().st_mtime
            boundary = started + committed_at - wall_started
            if not started <= boundary <= deadline:
                raise TimeoutError(f"Epoch {epoch} checkpoint exceeded its {cap / 3600:g}-hour wall-clock cap")
            if epoch == finished and pointer["completed"]:
                deadline = min(deadline, boundary + SETTLEMENT_SECONDS)
            else:
                deadline = boundary + cap
        committed = finished
        if code == 0:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Native trainer reported exit zero after the deadline for epoch {committed}")
            return code
        remaining = deadline - time.monotonic() - STOP_RESERVE_SECONDS
        if remaining <= 0:
            raise TimeoutError(f"Native trainer failed to settle with shutdown time left after epoch {committed}")
        try:
            process.wait(timeout=min(POLL_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            pass


def execute(queue, worktree, argv, formal, spec, visible_ascend):
    devices = spec["devices"]
    (queue / "npu_before.txt").write_text(start.host_health(devices), encoding="utf-8")
    environment = os.environ.copy()
    for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR",
                     "HCCL_IF_BASE_PORT", "ASCEND_VISIBLE_DEVICES"):
        environment.pop(variable, None)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    if visible_ascend is not None:
        environment["ASCEND_VISIBLE_DEVICES"] = visible_ascend
    record = dict(argv=argv, cwd=str(worktree), physical_npus=devices, epoch_hours=spec["hours"])
    with ExitStack() as files:
        stdout = files.enter_context((queue / "stdout.log").open("xb"))
        stderr = files.enter_context((queue / "stderr.log").open("xb"))
        if len(devices) == 2:
            import fcntl

            lease_path = queue.parent / f".hccl-{socket.gethostname()}-{HCCL_BASE_PORT}.lock"
            lease = files.enter_context(lease_path.open("ab"))
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            start.verify_host_port_block(HCCL_BASE_PORT)
            environment["HCCL_IF_BASE_PORT"] = str(HCCL_BASE_PORT)
            record["hccl_if_base_port"] = HCCL_BASE_PORT
        wall_started = time.time()
        started = time.monotonic()
        record["started_utc"] = start.utc()
        process = subprocess.Popen(argv, cwd=worktree, env=environment, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        record["pid"] = process.pid
        failure = None
        try:
            start.save_json(queue / "launch.json", record)
            start.save_json(queue / "actual_argv.json", argv)
            code = watch_training(process, formal, spec["hours"] * 3600, started, wall_started)
        except BaseException as error:
            failure = error
            code = start.ensure_stop(process)
        finally:
            if process.returncode is not None:
                temporary = queue / "exit_code.tmp"
                temporary.write_text(f"{process.returncode}\n", encoding="utf-8")
                temporary.replace(queue / "exit_code.txt")
            start.save_json(queue / "exit.json", dict(**record, ended_utc=start.utc(), exit_code=process.returncode,
                            error=str(failure) if failure else None, wall_seconds=time.monotonic() - started))
        if failure is not None:
            raise failure
        require(code == 0, f"Native successor actually exited {code}; see {queue / 'stderr.log'}")


def continue_training(queue, runtime, source, wrapper, *, dry_run):
    spec = start.SOURCES[source]
    milestone = 16 if source == start.FUSION else 8
    previous, launch, worktree, prefix, arguments = previous_request(runtime, source)
    exit_path = previous / "exit_code.txt"
    revision = verify_checkout(worktree, launch["trainer_git_sha"])
    script_sha = digest(worktree / TRAINER)
    code = await_first_exit(exit_path, launch["deadline_epoch"], wait=not dry_run)
    if code is None:
        return dict(status="waiting_for_first_epoch", exit_path=str(exit_path), deadline_epoch=launch["deadline_epoch"],
                    original_trainer=revision, next_output=str(queue / "formal"))
    first = dict(exit_path=str(exit_path), actual_exit_code=code, original_launch=launch, verified_trainer=revision)
    if not dry_run:
        start.save_json(queue / "first_epoch.json", first)
    require(code == 0, f"First epoch actually exited {code}: {exit_path}")
    original = previous / "formal"
    layout_record = start.read_json(original / "resumed-layout.json")
    layout = verify_layout(layout_record, spec, revision, script_sha, source, milestone)
    verify_inputs(arguments, layout_record["recipe"])
    incoming = dict(updates=launch["starting_updates"], exposure=launch["starting_updates"] * GLOBAL_BATCH,
                    epoch=1, next_batch=launch["starting_updates"])
    verify_restore_receipts(original, spec, revision, script_sha, incoming, launch["checkpoint_file_sha256"])
    checkpoint, first_curve = verify_formal(original, spec, milestone, completed=False)
    epoch_file_sha = digest(original / checkpoint["file"])
    formal = queue / "formal"
    arguments.pop("--stop-after-updates")
    arguments["--out"] = str(formal)
    argv = [*prefix, str(wrapper), "--restore-file-sha256", epoch_file_sha,
            *(token for pair in arguments.items() for token in pair), "--resume"]
    first.update(checkpoint=checkpoint["file"], checkpoint_file_sha256=epoch_file_sha, rank_layout=layout,
                 validation_support_sha256=digest(original / "validation-support.json"),
                 optimizer_updates=checkpoint["updates"], pairs=checkpoint["exposure"], validation=first_curve[0]["validation"])
    if dry_run:
        return dict(**first, status="ready", planned_argv=argv)

    start.save_json(queue / "first_epoch.json", first)
    clone_epoch(original, formal, epoch_file_sha)
    require(start.read_json(formal / "checkpoint.json") == checkpoint, "First-epoch checkpoint pointer changed during snapshot")
    verify_checkout(worktree, revision)
    require(digest(wrapper) == RESTORE_SHA, "Verified restore wrapper changed before launch")
    execute(queue, worktree, argv, formal, spec, layout[0].get("visible_ascend_devices"))
    final_layout = start.read_json(formal / "resumed-layout.json")
    verify_layout(final_layout, spec, revision, script_sha, source, milestone)
    require(final_layout["recipe"] == layout_record["recipe"], "Continuation recipe differs from first epoch")
    verify_restore_receipts(formal, spec, revision, script_sha, checkpoint, epoch_file_sha)
    final_checkpoint, curve = verify_formal(formal, spec, milestone, completed=True)
    require(curve[:1] == first_curve, "Continuation changed first-epoch validation or exposure")
    selected = start.read_json(formal / "selected.json")
    result = check_selection(curve, selected, spec, milestone)
    require(final_checkpoint["best"] == {key: selected[key] for key in ("epoch", "nll", "file")},
            "Final checkpoint disagrees with its selected candidate")
    selected_file = formal / result["selected_checkpoint"]
    return dict(**result, selected_checkpoint_path=str(selected_file), selected_checkpoint_file_sha256=digest(selected_file),
                confirmation_test_scored=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", choices=tuple(start.SOURCES), required=True)
    parser.add_argument("--dry-run", action="store_true", help="Inspect the registered state without claiming a lease")
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    checkout = Path(__file__).resolve().parents[2]
    wrapper = checkout / RESTORE
    revision = verify_checkout(checkout)
    require(digest(wrapper) == RESTORE_SHA, "Restore wrapper differs from the verified implementation")
    queue = runtime / SUCCESSORS / args.source
    if args.dry_run:
        print(json.dumps(continue_training(queue, runtime, args.source, wrapper, dry_run=True), indent=2), flush=True)
        return
    queue.mkdir(parents=True, exist_ok=False)
    registration = dict(source=args.source, pid=os.getpid(), registered_utc=start.utc(), controller_git_sha=revision,
                        restore_wrapper=str(wrapper), successor_output=str(queue / "formal"), status="waiting_for_first_epoch")
    start.save_json(queue / "registration.json", registration)
    print(json.dumps(registration), flush=True)

    def interrupt(sig, frame):
        raise InterruptedError(f"Received signal {sig}")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        result = continue_training(queue, runtime, args.source, wrapper, dry_run=False)
    except BaseException as error:
        start.save_json(queue / "settlement.json", dict(status="parked", ended_utc=start.utc(),
                        error_type=type(error).__name__, detail=str(error)))
        raise
    start.save_json(queue / "settlement.json", dict(**result, status="selection_complete", ended_utc=start.utc()))


if __name__ == "__main__":
    main()
