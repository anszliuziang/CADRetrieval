"""Deploy and collect the frozen eight-view baseline over one share4 connection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time

from collect_retrieval_benchmark_fill import REMOTE_RUN, sha256, verify_received
from finish_retrieval_benchmark_first_batch import write_status


SEED_SHA = "576325e3f0474731f19f75180ccf24d084993ef9"
ARMS = ("ct_deepcad", "l2", "l0", "l1", "l3")
GAP_SECONDS = 65


def remote_command(approved_sha: str, manifest_sha: str, *, pickup: bool = False) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", approved_sha) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        raise ValueError("Invalid reviewed Git or transfer manifest SHA")
    source = r"""set -euo pipefail
exec 3>&1 1>&2
umask 027
run='@REMOTE_RUN@'
rt=/home/ma-user/work/CADRetrieval_runtime
seed="$rt/worktrees/codex-retrieval-benchmark-duoduo-20261120"
work="$rt/worktrees/codex-retrieval-benchmark-duoduo-full-20261120"
stage="$run/duoduoclip_full"
control="$run/control/duoduoclip_full"
ref=refs/reviewed/codex/retrieval-benchmark-duoduo-full
"""
    if not pickup:
        source += r"""
test "$(git -C "$seed" rev-parse HEAD)" = '@SEED_SHA@'
test -z "$(git -C "$seed" status --porcelain)"
git -C "$seed" remote get-url origin | grep -Eq '^https://github\.com/anszliuziang/CADRetrieval(\.git)?$'
test ! -e "$work"
test ! -e "$control"
mkdir -- "$stage"
tar --keep-old-files --no-same-owner --no-same-permissions -xf - -C "$stage"
cd "$stage"
printf '%s  SHA256SUMS\n' '@MANIFEST_SHA@' | sha256sum --strict --check -
sha256sum --strict --status --check SHA256SUMS
printf 'render transport verified at %s\n' "$(date -u +%FT%TZ)"
git -C "$seed" bundle verify "$stage/work.bundle"
git -C "$seed" fetch --no-tags "$stage/work.bundle" "HEAD:$ref"
test "$(git -C "$seed" rev-parse "$ref")" = '@APPROVED_SHA@'
git -C "$seed" worktree add --detach "$work" '@APPROVED_SHA@'
bash "$work/experiments/scripts/launch_retrieval_benchmark_duoduoclip.sh" "$run" '@APPROVED_SHA@' 3>&-
"""
    source += r"""
cd "$stage"
printf '%s  SHA256SUMS\n' '@MANIFEST_SHA@' | sha256sum --strict --check -
test "$(cat "$control/implementation.sha")" = '@APPROVED_SHA@'
end=$(( $(date +%s) + 28800 ))
until [[ -s "$control/actual_exit" ]]; do
    if (( $(date +%s) >= end )); then printf 'CPU score collection timed out\n'; exit 4; fi
    sleep 20
done
test "$(cat "$control/actual_exit")" = 0
cd "$run"
files=(control/duoduoclip_full/actual_exit control/duoduoclip_full/argv.txt
       control/duoduoclip_full/implementation.sha control/duoduoclip_full/stdout.log
       control/duoduoclip_full/stderr.log control/duoduoclip_full/launcher.pid
       control/duoduoclip_full/started.txt duoduoclip_full/SHA256SUMS
       duoduoclip_full/render.actual_exit.json duoduoclip_full/renders/render_summary.json
       external/duoduoclip/duoduoclip_demo.json)
for stem in gallery @TEXT_AXES@; do
    files+=("external/duoduoclip/$stem.npy" "external/duoduoclip/${stem}_status.jsonl"
            "external/duoduoclip/${stem}_meta.json")
done
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/TRANSFER_SHA256SUMS"; rmdir -- "$scratch"' EXIT
sha256sum -- "${files[@]}" > "$scratch/TRANSFER_SHA256SUMS"
tar -cf - "${files[@]}" -C "$scratch" TRANSFER_SHA256SUMS >&3
"""
    for key, value in (
        ("@REMOTE_RUN@", REMOTE_RUN), ("@SEED_SHA@", SEED_SHA), ("@MANIFEST_SHA@", manifest_sha),
        ("@APPROVED_SHA@", approved_sha), ("@TEXT_AXES@", " ".join("texts_" + arm for arm in ARMS)),
    ):
        source = source.replace(key, value)
    return source


def waiting_for_connection(runtime: Path, pack: Path, output: Path, deadline: float,
                           prior_ssh: float) -> None:
    while time.time() < deadline:
        snapshots = [json.loads(path.read_bytes()) for path in
                     (runtime / "control").glob("finish_first_batch*/current.json")]
        state = max(snapshots, key=lambda snapshot: snapshot["utc_epoch"])
        last_start = None
        if state.get("phase") == "collect_text2cad_l2" and state["state"] == "running":
            argv = state["argv"]
            receipt_path = Path(argv[argv.index("--output") + 1]) / "connection_start.json"
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_bytes())
                if (receipt["kind"] != "text2cad" or receipt["conditions"] != ["l2"]
                        or receipt["started_epoch"] < state["utc_epoch"]):
                    raise ValueError("The L2 connection receipt does not match the active collector")
                last_start = receipt["started_epoch"]
        elif state["state"] == "completed":
            last_start = state["utc_epoch"]
        if (pack.is_file() and last_start is not None
                and time.time() >= max(last_start, prior_ssh) + GAP_SECONDS):
            return
        if state["state"] in {"failed", "partial_external"}:
            raise RuntimeError("Wait for recovery of the first-batch collector")
        write_status(output / "current.json", state="waiting", pack_ready=pack.is_file(),
                     primary_phase=state.get("phase"))
        time.sleep(30)
    raise TimeoutError("The verified render archive or final first-batch connection is pending")


def collect(runtime: Path, output: Path, approved_sha: str, pickup_from: Path | None) -> int:
    plan = runtime / "control/duoduo_full_payload"
    pack = plan / "pack_receipt.json"
    prior_ssh = (json.loads((pickup_from / "connection_start.json").read_bytes())["started_epoch"]
                 if pickup_from else 0)
    waiting_for_connection(runtime, pack, output, time.time() + 9 * 3600, prior_ssh)
    receipt = json.loads(pack.read_bytes())
    if (receipt["all_renders"] != 8032 or receipt["valid_renders"] < 2
            or receipt["files"] != 4 + 8 * receipt["valid_renders"]
            or sha256(plan / "SHA256SUMS") != receipt["manifest_sha256"]
            or (not pickup_from and sha256(plan / "payload.tar") != receipt["tar_sha256"])):
        raise ValueError("Eight-view transport does not match the verified local pack receipt")
    script = remote_command(approved_sha, receipt["manifest_sha256"], pickup=pickup_from is not None)
    (output / "remote_command.sh").write_text(script, encoding="ascii", newline="\n")
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=2",
            "yunnao-share4", "bash -c " + shlex.quote(script)]
    source = plan / "payload.tar" if pickup_from is None else Path(os.devnull)
    with source.open("rb") as src, (output / "artifact.tar").open("xb") as dst, \
            (output / "ssh.stderr.log").open("xb") as err:
        with subprocess.Popen(argv, stdin=src, stdout=dst, stderr=err) as result:
            started = time.time()
            write_status(output / "connection_start.json", started_epoch=started, ssh_pid=result.pid)
            write_status(output / "current.json", state="uploading_or_scoring", started_epoch=started)
            result.wait()
    record = {"actual_ssh_exit": result.returncode, "implementation_sha": approved_sha,
              "started_epoch": started, "conditions": list(ARMS), "kind": "duoduoclip",
              "pickup": pickup_from is not None}
    write_status(output / "ssh_exit.json", **record)
    if result.returncode == 0:
        raw = output / "raw"
        raw.mkdir()
        with tarfile.open(output / "artifact.tar", "r:") as archive:
            archive.extractall(raw, filter="data")
        record.update(verify_received(raw))
        if ((raw / "control/duoduoclip_full/actual_exit").read_text().strip() != "0"
                or (raw / "control/duoduoclip_full/implementation.sha").read_text().strip() != approved_sha
                or sha256(raw / "duoduoclip_full/SHA256SUMS") != receipt["manifest_sha256"]):
            raise ValueError("Returned score evidence does not certify this execution and its inputs")
    write_status(output / "pull_receipt.json", **record)
    write_status(output / "current.json", state="completed" if result.returncode == 0 else "failed",
                 actual_ssh_exit=result.returncode)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-sha", required=True)
    parser.add_argument("--pickup-from", type=Path, help="Prior connection evidence for read-only recovery")
    args = parser.parse_args()
    args.output.mkdir()
    try:
        result = collect(args.runtime, args.output, args.approved_sha, args.pickup_from)
    except (OSError, RuntimeError, ValueError) as error:
        write_status(args.output / "failure.json", error=str(error))
        raise
    raise SystemExit(result)


if __name__ == "__main__":
    main()
