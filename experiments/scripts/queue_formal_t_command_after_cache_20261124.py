"""Run the three formal T command-tower seeds after node 4's pooled BERT merge."""

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep, time

CODE_SHA = "f8288dd16de36e8f27aad371b2008dce4c35c3bb"
AXIS_SHA = "0be88b85373f9dd558f46432aebff98623f02ae9daea49467c7842875fc07695"
SEED_CHIPS = ((17, 0), (29, 1), (43, 2))
CACHE_WORKER_SECONDS = 32400
CACHE_MERGE_SECONDS = 1200
TRAIN_SECONDS = 21600
AUDIT_SECONDS = 1200
KILL_GRACE = 15
POLL_SECONDS = 15
SOURCE = "t2cad-v1.0"
TRAIN_UIDS, TRAIN_PAIRS, TRAIN_UPDATES = 151_454, 605_816, 18_932
VALIDATION_TARGETS, VALIDATION_GALLERY, VALIDATION_QUERIES = 512, 2_048, 2_048
CACHE_SHAPE = [446_461, 1024]
BERT_REV = "6da4b6a26a1877e173fca3225479512db81a5e5b"
DATASET_REV = "422d4c93198c3cd98d5c2092c0ab214b3a76a68d"
PRODUCER_NAME = "idea01_four_tables_T_pooled_bert_20261124"
QUEUE_NAME = "idea01_four_tables_T_command_after_cache_20261124"
TRAIN_MODULE = "experiments.scripts.run_in_domain_pilot_retrievers"
AUDIT_MODULE = "experiments.scripts.audit_formal_command_training"


def utc(timestamp=None):
    value = time() if timestamp is None else timestamp
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_directory(path):
    path.mkdir()
    sync_dir(path.parent)


def save_text(path, content):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".receipt-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def save_json(path, record):
    save_text(path, json.dumps(record, allow_nan=False, indent=2) + "\n")


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, timeout=30).strip()


def check_checkout(repo):
    if (git(repo, "rev-parse", "HEAD") != CODE_SHA or git(repo, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD"
            or git(repo, "--no-optional-locks", "status", "--porcelain", "--untracked-files=all")):
        raise ValueError(f"Expected a clean detached checkout at {CODE_SHA}: {repo}")


def successful_exit(path, start):
    try:
        with path.open(encoding="utf-8") as stream:
            code = stream.read()
            ended = os.fstat(stream.fileno()).st_mtime
    except FileNotFoundError:
        return None
    if not code.endswith("\n"):
        return None  # The existing shell producer writes its exit file directly.
    if code.strip() != "0" or ended < start:
        raise RuntimeError(f"Invalid or unsuccessful producer exit {code.strip()!r} at {path}")
    return ended


def wait_for_producer(producer):
    control = producer / "full_workers"
    start_record = datetime.fromisoformat((control / "npu_started").read_text(encoding="utf-8").strip())
    if start_record.tzinfo is None or start_record.timestamp() > time():
        raise ValueError("Invalid producer npu_started timestamp")
    started = start_record.timestamp()
    utc_offset = time() - monotonic()
    worker_deadline = started + CACHE_WORKER_SECONDS
    while True:
        producer_ended = successful_exit(control / "exit_code.txt", started)
        worker_exits = [successful_exit(control / f"chip_{chip}/exit_code.txt", started) for chip in range(8)]
        if any(value is not None and value > worker_deadline for value in worker_exits):
            raise TimeoutError("A BERT worker exited after the nine-hour producer window")
        if all(value is not None for value in worker_exits):
            last_worker = max(worker_exits)
            deadline = last_worker + CACHE_MERGE_SECONDS
            if producer_ended is not None:
                if not last_worker <= producer_ended <= deadline:
                    raise TimeoutError("The successful BERT producer exit falls outside its merge window")
                return {"npu_started_utc": utc(started), "worker_exit_utc": [utc(value) for value in worker_exits],
                        "producer_exit_utc": utc(producer_ended), "producer_exit_path": str(control / "exit_code.txt")}
        else:
            if producer_ended is not None:
                raise RuntimeError("The BERT producer completed without all eight successful worker receipts")
            deadline = worker_deadline
        remaining = deadline - utc_offset - monotonic()
        if remaining <= 0:
            raise TimeoutError("The BERT producer did not settle within its worker and merge windows")
        sleep(min(POLL_SECONDS, remaining))


def verified_cache(producer):
    axis_raw = (producer / "axis/receipt.json").read_bytes()
    if sha256(axis_raw).hexdigest() != AXIS_SHA:
        raise ValueError("The prepared BERT axis receipt does not match its registered SHA-256")
    axis = json.loads(axis_raw)
    merged = producer / "merged"
    raw = (merged / "receipt.json").read_bytes()
    receipt = json.loads(raw)
    if (receipt["schema"] != "idea01-t-command-bert-cache-v1" or receipt["axis_sha256"] != AXIS_SHA
            or receipt["matrix_shape"] != CACHE_SHAPE or receipt["matrix_dtype"] != "float32"
            or any(receipt[key] != axis[key] for key in ("policy", "model", "tokenizer", "input_histogram"))
            or receipt["files_sha256"]["text_index.jsonl"] != axis["files_sha256"]["text_index.jsonl"]):
        raise ValueError("The merged BERT cache identity differs from the registered training axis")
    files = receipt["files_sha256"]
    for name in ("pooled.npy", "text_index.jsonl"):
        if not re.fullmatch(r"[0-9a-f]{64}", files[name]) or (merged / name).stat().st_size == 0:
            raise ValueError(f"The merged BERT cache has an invalid file receipt for {name}")
    # The pinned runner hashes both files and checks the actual array and exact training texts.
    return {"path": str(merged.resolve()), "receipt_sha256": sha256(raw).hexdigest(), "axis_sha256": AXIS_SHA,
            "matrix_shape": CACHE_SHAPE, "matrix_dtype": "float32"}


def training_command(runtime, output, merged, seed):
    python = runtime / "envs/cadretrieval-910a-cann80-torch240/bin/python"
    manifests = runtime / "four_tables_inputs_20261124/prior_share4/runs/idea01_in_domain_pilot_20261123/manifests"
    archive = runtime / "datasets/Text2CAD-v1.0" / DATASET_REV / "cad_seq.zip"
    bert = runtime / "models/bert-large-uncased" / BERT_REV
    store = runtime / "runs/idea01_four_tables_effective_inputs_20261124/deepcad_command"
    for required in (python, manifests / "manifest_receipt.json", archive, bert / "config.json", store / "summary.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    return [str(python), "-B", "-u", "-m", TRAIN_MODULE, "--manifest-root", str(manifests), "--source", SOURCE,
            "--method", "dual-tower", "--official-zip", str(archive), "--shared-feature-store", str(store),
            "--train-split", "full", "--require-full-validation-text", "--seed", str(seed), "--precision", "fp32",
            "--attention", "eager", "--batch-size", "32", "--eval-batch-size", "32", "--command-convolution", "linear",
            "--expected-train-uids", str(TRAIN_UIDS), "--expected-targets", str(VALIDATION_TARGETS),
            "--expected-gallery", str(VALIDATION_GALLERY), "--bert-dir", str(bert), "--training-bert-cache", str(merged),
            "--device", "npu:0", "--output", str(output)]


def bounded_command(argv, seconds):
    return ["timeout", "--signal=TERM", f"--kill-after={KILL_GRACE}s", f"{seconds - KILL_GRACE}s", *argv]


async def free_device(chip, probe_lock):
    while True:
        async with probe_lock:
            health = await asyncio.to_thread(subprocess.check_output, ["npu-smi", "info"], text=True, timeout=20)
        if re.search(rf"^\|\s+{chip}\s+910ProA\s+\|\s+OK\s+\|", health, re.MULTILINE) is None:
            raise RuntimeError(f"No healthy 910ProA reading for physical NPU {chip}")
        if re.search(rf"^\|\s+No running processes found in NPU {chip}\s+\|", health, re.MULTILINE):
            return health
        await asyncio.sleep(POLL_SECONDS)


def phase_environment(chip):
    environment = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "ASCEND_RT_VISIBLE_DEVICES"):
        environment.pop(key, None)
    environment.update(PYTHONNOUSERSITE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0", HF_HUB_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       ACL_PRECISION_MODE="must_keep_origin_dtype")
    if chip is not None:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(chip)
    return environment


async def stop_phase(process):
    for sig, seconds in ((signal.SIGTERM, KILL_GRACE), (signal.SIGKILL, 5)):
        if process.returncode is not None:
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), seconds)
        except TimeoutError:
            continue
    return process.returncode


async def run_phase(repo, folder, argv, seed, chip, seconds):
    command = bounded_command(argv, seconds)
    started = monotonic()
    record = {"seed": seed, "physical_chip": chip, "git_sha": CODE_SHA,
              "argv": command, "started_utc": utc(), "wall_limit_seconds": seconds}
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=repo, env=phase_environment(chip), stdout=stdout, stderr=stderr, start_new_session=True,
        )
        record["pid"] = process.pid
        interrupted = True
        code = None
        try:
            save_json(folder / "launch.json", record)
            code = await process.wait()
            interrupted = False
        except BaseException:
            code = await stop_phase(process)
            raise
        finally:
            save_json(folder / "exit.json", {**record, "exit_code": code, "interrupted": interrupted,
                                             "ended_utc": utc(), "wall_seconds": monotonic() - started})
            if code is not None:
                save_text(folder / "exit_code.txt", f"{code}\n")
    if code != 0:
        raise RuntimeError(f"Seed {seed} {folder.name} exited {code}: {folder}")


def read_json(path):
    return json.loads(path.read_bytes())


def check_training(folder, chip, cache):
    result = folder / "result"
    run = read_json(result / "run.json")
    config = read_json(result / "config.json")
    actual = run["training"]
    counts = (actual["actual_training_uids"], actual["actual_training_pairs"], actual["actual_updates"],
              actual["actual_distinct_training_texts"], run["validation_queries_total"], run["validation_gallery_total"])
    if (counts != (TRAIN_UIDS, TRAIN_PAIRS, TRAIN_UPDATES, CACHE_SHAPE[0], VALIDATION_QUERIES, VALIDATION_GALLERY)
            or run["training_cad_coverage"] != {"valid": TRAIN_UIDS, "total": TRAIN_UIDS}
            or config["visible_devices"].get("ASCEND_RT_VISIBLE_DEVICES") != str(chip)
            or config["per_device_evaluation_batch"] != 32 or config["attention"] != "eager"
            or config["command_convolution"] != "linear"
            or any(config["training_bert_cache"][key] != value for key, value in cache.items())):
        raise ValueError(f"Seed coverage or frozen-cache recipe does not match the queue admission: {folder}")


async def run_seed(repo, folder, argv, seed, chip, cache, probe_lock):
    create_directory(folder)
    try:
        save_text(folder / "npu_before.txt", await free_device(chip, probe_lock))
        await run_phase(repo, folder, argv, seed, chip, TRAIN_SECONDS)
        check_training(folder, chip, cache)
        audit = folder / "audit"
        create_directory(audit)
        verified = audit / "recomputed.json"
        audit_argv = [*argv[:4], AUDIT_MODULE, "--run", str(folder / "result"),
                      "--manifest-root", argv[argv.index("--manifest-root") + 1], "--source", SOURCE,
                      "--exit-record", str(folder / "exit_code.txt"), "--expected-source-sha", CODE_SHA,
                      "--seed", str(seed), "--output", str(verified)]
        await run_phase(repo, audit, audit_argv, seed, None, AUDIT_SECONDS)
        coverage = read_json(verified)
        if coverage["all_passed"] is not True or coverage["validation"]["rank_rows_verified"] != VALIDATION_QUERIES:
            raise ValueError(f"Seed {seed} is missing independent full validation coverage")
        save_json(folder / "coverage.json", {**coverage, "physical_chip": chip, "cache": cache})
        return {"seed": seed, "state": "completed", "coverage": str(folder / "coverage.json")}
    except Exception as exc:  # noqa: BLE001 - One seed failure must not cancel the other seeds.
        failure = {"seed": seed, "state": "failed", "error": f"{type(exc).__name__}: {exc}"}
        save_json(folder / "failure.json", failure)
        return failure


async def run_seeds(repo, queue, commands, cache):
    probe_lock = asyncio.Lock()
    return await asyncio.gather(*(run_seed(repo, queue / f"seed_{seed}", commands[seed], seed, chip, cache, probe_lock)
                                  for seed, chip in SEED_CHIPS))


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="Node 4's CADRetrieval_runtime directory")
    parser.add_argument("--dry-run", action="store_true", help="Check the checkout and input paths and print the plan")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    repo = runtime / "worktrees/cad-fleet-f8288dd"
    producer = runtime / "runs" / PRODUCER_NAME
    queue = runtime / "runs" / QUEUE_NAME
    check_checkout(repo)
    for program in ("timeout", "npu-smi"):
        if shutil.which(program) is None:
            raise FileNotFoundError(f"Required server utility is unavailable: {program}")
    commands = {seed: training_command(runtime, queue / f"seed_{seed}/result", producer / "merged", seed)
                for seed, _ in SEED_CHIPS}
    registration = {"registered_utc": utc(), "pid": os.getpid(), "git_sha": CODE_SHA, "checkout": str(repo),
                    "producer": str(producer), "queue": str(queue), "axis_sha256": AXIS_SHA,
                    "worker_time_cap_seconds": CACHE_WORKER_SECONDS, "merge_time_cap_seconds": CACHE_MERGE_SECONDS,
                    "train_time_cap_seconds_per_seed": TRAIN_SECONDS, "audit_time_cap_seconds_per_seed": AUDIT_SECONDS,
                    "test_scoring_executed": False,
                    "seeds": [{"seed": seed, "physical_chip": chip, "argv": bounded_command(commands[seed], TRAIN_SECONDS)}
                              for seed, chip in SEED_CHIPS]}
    if args.dry_run:
        print(json.dumps(registration, indent=2))
        return 0

    os.umask(0o027)
    signal.signal(signal.SIGTERM, interrupt)
    create_directory(queue)  # Exclusive run name prevents a duplicate invocation from launching any seed.
    save_json(queue / "registration.json", registration)
    try:
        producer_exits = wait_for_producer(producer)
        cache = verified_cache(producer)
        check_checkout(repo)
        save_json(queue / "cache_ready.json", {**producer_exits, **cache, "admitted_utc": utc()})
        results = asyncio.run(run_seeds(repo, queue, commands, cache))
        completed = all(result["state"] == "completed" for result in results)
        settlement = {"state": "completed" if completed else "failed", "seeds": results}
        code = 0 if completed else 1
    except KeyboardInterrupt:
        settlement, code = {"state": "interrupted"}, 130
    except Exception as exc:  # noqa: BLE001 - Preserve a settlement for unexpected controller failures.
        settlement, code = {"state": "failed", "error": f"{type(exc).__name__}: {exc}"}, 1
    save_json(queue / "settlement.json", {**settlement, "ended_utc": utc(), "exit_code": code})
    save_text(queue / "exit_code.txt", f"{code}\n")
    print(json.dumps(settlement), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
