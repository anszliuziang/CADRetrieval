"""Accept the first benchmark batch and execute the bounded L1/L3 NPU probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

from collect_retrieval_benchmark_fill import REMOTE_RUN, sha256, verify_received
from finish_retrieval_benchmark_first_batch import write_status


ROOT = Path(__file__).resolve().parents[2]
AUTH_PATH = "experiments/configs/idea01_retrieval_benchmark_fill_run_authorization.json"
FIRST_AUTH = "fc71f871a8da2aa78bfcff3714ff3b5afa9d546d"
SCORER = "6cd791312eefa5e575b0e56a39553fa9f92fbd3b"
CONDITIONS = ["l1", "l3"]
PROBE_PAIRS = 2560
CARD_SECONDS = 900
OUTPUT_DIRECTORY = "probe_native_tail"
POLL_SECONDS = 45
SSH_GAP_SECONDS = 65


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def git_json(sha: str) -> dict:
    return json.loads(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{sha}:{AUTH_PATH}"]))


def verify_authorization(sha: str, bundle: Path) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise ValueError("Expected the full reviewed authorization SHA")
    expected = git_json(FIRST_AUTH)
    del expected["probe_result_sha256"]
    expected.update(mode="probe", output_directory=OUTPUT_DIRECTORY, approved_sha=SCORER,
                    conditions=CONDITIONS, max_pairs=PROBE_PAIRS, max_card_seconds=CARD_SECONDS, max_devices=1)
    auth = git_json(sha)
    if auth != expected:
        raise ValueError("The native follow-up authorization changes the planned scoring or resource limits")
    heads = subprocess.check_output(["git", "-C", str(ROOT), "bundle", "list-heads", str(bundle)],
                                   text=True).splitlines()
    if heads != [f"{sha} HEAD"]:
        raise ValueError("The transport bundle does not bind the reviewed authorization")
    return auth


def wait_for_first_batch(run: Path, control: Path, deadline: float) -> Path:
    while time.time() < deadline:
        states = [(p, read_json(p)) for p in (run / "control").glob("finish_first_batch*/current.json")]
        if not states:
            write_status(control / "current.json", state="waiting", phase="first_batch_acceptance")
            time.sleep(POLL_SECONDS)
            continue
        path, state = max(states, key=lambda item: item[1]["utc_epoch"])
        if state["state"] == "completed":
            return path.parent
        if state["state"] in ("failed", "partial_external"):
            raise RuntimeError(f"The first batch has unresolved acceptance evidence at {path}")
        write_status(control / "current.json", state="waiting", phase="first_batch_acceptance",
                     upstream_phase=state.get("phase"))
        time.sleep(POLL_SECONDS)
    raise TimeoutError("The bounded follow-up window expired before first-batch acceptance")


def wait_for_ssh_gap(run: Path, first_finished: float, deadline: float) -> None:
    duoduo = run / "received/duoduoclip_full"
    duoduo_launcher = run / "control/duoduo_full_collect_launcher.json"
    while time.time() < deadline:
        # The already queued collector has priority if both wait on first-batch completion.
        predecessor = first_finished
        if duoduo_launcher.is_file() and not (duoduo / "connection_start.json").is_file():
            failure = duoduo / "failure.json"
            if not failure.is_file():
                time.sleep(min(POLL_SECONDS, max(0, deadline - time.time())))
                continue
            predecessor = max(predecessor, read_json(failure)["utc_epoch"])
        paths = list((run / "received").glob("*/connection_start.json"))
        paths.extend((run / "control").glob("share4_*_receipt.json"))
        recorded = [read_json(p).get("started_epoch", 0) for p in paths]
        last_start = max([predecessor, *recorded])
        remaining = last_start + SSH_GAP_SECONDS - time.time()
        if remaining <= 0:
            return
        time.sleep(min(POLL_SECONDS, remaining))
    raise TimeoutError("No share4 connection can start inside the reviewed window")


def remote_command(auth_sha: str, bundle_sha: str, ledger_sha: str, deadline: int) -> str:
    if (re.fullmatch(r"[0-9a-f]{40}", auth_sha) is None
            or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (bundle_sha, ledger_sha))
            or type(deadline) is not int or deadline <= 0):
        raise ValueError("Invalid transport or acceptance pins")
    command = r"""set -eo pipefail
exec 3>&1 1>&2
umask 027
rt=/home/ma-user/work/CADRetrieval_runtime
run='@RUN@'
seed="$rt/worktrees/codex-retrieval-benchmark-fill-20261120"
checkout="$rt/worktrees/codex-retrieval-benchmark-native-probe-20261120"
control="$run/control/native_tail_probe"
out="$run/probe_native_tail"
ref=refs/remotes/origin/codex/retrieval-benchmark-fill
py=/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python
test "$(date +%s)" -lt '@DEADLINE@'
test "$(cat "$run/control/score/controller.actual_exit")" = 0
printf '%s  %s\n' '@LEDGER_SHA@' "$run/full/ledger.json" | sha256sum --strict --check -
test "$(git -C "$seed" rev-parse HEAD)" = '@SCORER@'
test "$(git -C "$seed" rev-parse "$ref")" = '@FIRST_AUTH@'
test -z "$(git -C "$seed" status --porcelain)"
test ! -e "$checkout"
test ! -e "$out"
mkdir -- "$control"
cat > "$control/authorization.bundle"
printf '%s  %s\n' '@BUNDLE_SHA@' "$control/authorization.bundle" | sha256sum --strict --check -
git clone --no-checkout --no-tags --shared -- "$seed" "$checkout"
git -C "$checkout" remote set-url origin https://github.com/anszliuziang/CADRetrieval.git
git -C "$checkout" bundle verify "$control/authorization.bundle"
git -C "$checkout" fetch --no-tags "$control/authorization.bundle" "+HEAD:$ref"
test "$(git -C "$checkout" rev-parse "$ref")" = '@AUTH_SHA@'
git -C "$checkout" checkout --detach '@SCORER@'
test -z "$(git -C "$checkout" status --porcelain)"
test -x "$py"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -- "$control/hf" "$control/tmp" "$control/xdg"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GIT_TERMINAL_PROMPT=0
export HF_HOME="$control/hf" XDG_CACHE_HOME="$control/xdg" TMPDIR="$control/tmp"
unset ASCEND_RT_VISIBLE_DEVICES
args=(
    "$py" -B -u "$checkout/experiments/scripts/run_retrieval_benchmark_text2cad.py" controller
    --mode probe --auth-sha '@AUTH_SHA@'
    --input "$run/frozen" --out "$out" --replay "$run/control/historical_replay.json"
    --checkpoint "$rt/models/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/Text2CAD_1.0.pth"
    --bert-dir "$rt/models/bert-large-uncased/6da4b6a26a1877e173fca3225479512db81a5e5b"
    --cad-zip "$rt/datasets/Text2CAD-v1.0/422d4c93198c3cd98d5c2092c0ab214b3a76a68d/cad_seq.zip"
)
printf '%q ' "${args[@]}" > "$control/launch_command.txt"
printf '\n' >> "$control/launch_command.txt"
date -u +%FT%TZ > "$control/launch_time.txt"
nohup bash -c '
    set +e
    result_file=$1
    code=$2
    shift 2
    "$@"
    result=$?
    if [[ $result == 0 && -n "$(git -C "$code" status --porcelain)" ]]; then
        result=2
    fi
    printf "%s\n" "$result" > "$result_file.tmp"
    mv -- "$result_file.tmp" "$result_file"
    exit "$result"
' tail_probe "$control/controller.actual_exit" "$checkout" "${args[@]}" 3>&- \
    </dev/null >"$control/stdout.log" 2>"$control/stderr.log" &
printf '%s\n' "$!" > "$control/launcher.pid"
until test -s "$control/controller.actual_exit"; do
    if (( $(date +%s) > @DEADLINE@ + 60 )); then exit 4; fi
    sleep 20
done
cd "$run"
files=(control/native_tail_probe/controller.actual_exit control/native_tail_probe/launcher.pid
       control/native_tail_probe/stdout.log control/native_tail_probe/stderr.log
       control/native_tail_probe/launch_command.txt control/native_tail_probe/launch_time.txt)
if [[ "$(cat "$control/controller.actual_exit")" == 0 ]]; then
    files+=(probe_native_tail/{controller_exit.json,gallery.jsonl,last_availability.json,ledger.json,probe_result.json,run_authorization.json}
            probe_native_tail/probe_exit_*.json probe_native_tail/probe_worker0_*.json
            probe_native_tail/probe_shard0_*.launch.json probe_native_tail/probe_shard0_*.stdout.log
            probe_native_tail/probe_shard0_*.stderr.log)
fi
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/TRANSFER_SHA256SUMS"; rmdir -- "$scratch"' EXIT
sha256sum -- "${files[@]}" > "$scratch/TRANSFER_SHA256SUMS"
tar -cf - "${files[@]}" -C "$scratch" TRANSFER_SHA256SUMS >&3
"""
    for key, value in {
        "@RUN@": REMOTE_RUN, "@AUTH_SHA@": auth_sha, "@FIRST_AUTH@": FIRST_AUTH, "@SCORER@": SCORER,
        "@BUNDLE_SHA@": bundle_sha, "@LEDGER_SHA@": ledger_sha, "@DEADLINE@": str(deadline),
    }.items():
        command = command.replace(key, value)
    return command


def verify_probe(raw: Path, auth_sha: str, auth: dict) -> dict:
    from run_retrieval_benchmark_text2cad import project_capacity

    status = int((raw / "control/native_tail_probe/controller.actual_exit").read_text().strip())
    if status != 0:
        raise RuntimeError(f"The follow-up probe controller exited {status}")
    source = raw / OUTPUT_DIRECTORY
    conclusion = read_json(source / "controller_exit.json")
    report = read_json(source / "probe_result.json")
    ledger = read_json(source / "ledger.json")
    if (conclusion["exit_code"] != 0 or conclusion["conditions"] != ["probe"]
            or conclusion["authorization_sha"] != auth_sha
            or conclusion["implementation_sha"] != SCORER or read_json(source / "run_authorization.json") != auth):
        raise ValueError("The completed probe does not match its authorization")
    exits = [read_json(path)["exits"] for path in source.glob("probe_exit_*.json")]
    if exits != [[0]]:
        raise ValueError("The probe does not have exactly one successful worker exit")
    invocations = ledger["invocations"]
    seconds = ledger["card_seconds_spent"]
    if (ledger["approved_sha"] != SCORER or ledger["pairs_charged"] != PROBE_PAIRS
            or ledger["max_pairs"] != PROBE_PAIRS or ledger["max_card_seconds"] != CARD_SECONDS
            or len(invocations) != 1 or not math.isfinite(seconds) or not 0 <= seconds <= CARD_SECONDS):
        raise ValueError("The follow-up probe ledger exceeds or changes its approved scope")
    item = invocations[0]
    if (item["finalized"] is not True or item["exit_code"] != 0 or item["cards_used"] != 1
            or item["pairs_charged"] != PROBE_PAIRS or item["card_seconds"] != seconds
            or not 0 < item["started_epoch"] <= item["ended_epoch"] <= auth["deadline_epoch"]):
        raise ValueError("Missing successful settlement of the probe worker")
    if (report["passed"] is not True or report["implementation_sha"] != SCORER
            or report["conditions"] != CONDITIONS or report["pairs_charged"] != PROBE_PAIRS
            or report["freeze_sha256"] != auth["freeze_sha256"] or report["replay_sha256"] != auth["replay_sha256"]):
        raise ValueError("The native-window probe result failed its frozen checks")
    measurements = report["measurements"]
    if [row["arm"] for row in measurements] != ["l0_history"] * 4 + ["l1"] * 3 + ["l3"] * 3:
        raise ValueError("The probe does not contain all prescribed historical, L1 and L3 cases")
    for row in measurements:
        if row["pairs_per_pass"] != 128 or type(row["cache_bitwise_equal"]) is not bool:
            raise ValueError("Invalid probe case scope")
        for field in ("cached_batch_seconds", "uncached_batch_seconds"):
            values = row[field]
            if len(values) != 8 or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("Incomplete or invalid probe batch timings")
        if row["arm"] != "l0_history":
            tokens = row["text_tokens"]
            count = tokens["token_count"]
            if (not 0 < count or tokens["effective_token_count"] != min(count, 512)
                    or tokens["truncated"] is not (count > 512)):
                raise ValueError("Invalid probe token-window evidence")
    if not any(row["text_tokens"]["truncated"] for row in measurements[-3:]):
        raise ValueError("L3 needs an actual overlength representative")
    if report["cache_passed"] is not (report["cache_hits"] > 0 and
                                     all(row["cache_bitwise_equal"] for row in measurements)):
        raise ValueError("The recorded cache verdict is inconsistent")
    projections = [project_capacity(report, CONDITIONS, devices) for devices in (1, 2, 3, 4)]
    if any(not math.isfinite(row[key]) or row[key] <= 0
           for row in projections for key in ("wall_seconds", "card_seconds")):
        raise ValueError("Invalid measured L1/L3 throughput projection")
    return {"probe_sha256": sha256(source / "probe_result.json"), "pairs": PROBE_PAIRS, "card_seconds": seconds,
            "cache_bitwise_equal": report["cache_passed"], "projections": projections}


def run_once(run: Path, control: Path, bundle: Path, auth_sha: str) -> None:
    from certify_retrieval_benchmark_first_batch import certify

    auth = verify_authorization(auth_sha, bundle)
    logs = wait_for_first_batch(run, control, auth["deadline_epoch"])
    certificate = certify(run, logs)
    write_status(control / "first_batch_accepted.json", **certificate)
    command = remote_command(auth_sha, sha256(bundle), certificate["ledger_sha256"], auth["deadline_epoch"])
    (control / "remote_command.sh").write_text(command, encoding="ascii", newline="\n")
    write_status(control / "current.json", state="waiting", phase="share4_connection_window")
    wait_for_ssh_gap(run, read_json(logs / "current.json")["utc_epoch"], auth["deadline_epoch"])
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=2", "yunnao-share4", "bash -c " + shlex.quote(command)]
    with bundle.open("rb") as source, (control / "artifact.tar").open("xb") as archive, \
            (control / "ssh.stderr.log").open("xb") as err:
        with subprocess.Popen(argv, stdin=source, stdout=archive, stderr=err) as process:
            write_status(control / "connection_start.json", started_epoch=time.time(), ssh_pid=process.pid)
            write_status(control / "current.json", state="running", phase="l1_l3_probe")
            process.wait()
    write_status(control / "ssh_exit.json", actual_exit=process.returncode)
    if process.returncode != 0:
        raise RuntimeError(f"The probe transport exited {process.returncode}; inspect launch receipts before recovery")
    raw = control / "raw"
    raw.mkdir()
    with tarfile.open(control / "artifact.tar", "r:") as archive:
        archive.extractall(raw, filter="data")
    coverage = verify_received(raw)
    write_status(control / "pull_receipt.json", actual_exit=0, **coverage)
    result = verify_probe(raw, auth_sha, auth)
    write_status(control / "probe_accepted.json", authorization_sha=auth_sha, **result)
    write_status(control / "current.json", state="completed", phase="l1_l3_probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--authorization-sha", required=True)
    args = parser.parse_args()
    control = args.runtime / "control/continue_native_probe"
    control.mkdir()
    try:
        run_once(args.runtime, control, args.bundle, args.authorization_sha)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.SubprocessError, tarfile.TarError) as error:
        write_status(control / "current.json", state="failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
