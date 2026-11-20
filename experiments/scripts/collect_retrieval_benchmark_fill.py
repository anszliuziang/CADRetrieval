"""Collect completed retrieval artifacts through a single read-only SSH stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time


REMOTE_RUN = "/home/ma-user/work/CADRetrieval_runtime/runs/idea01_retrieval_benchmark_fill_20261120"
CORE_EXTERNAL = ("clip", "openshape", "mmcad_v2")
CONDITIONS = ("ct_deepcad", "l2", "l1", "l3")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_received(root: Path) -> dict:
    manifest = root / "TRANSFER_SHA256SUMS"
    files = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if (not re.fullmatch(r"[a-f0-9]{64}", checksum) or relative.is_absolute()
                or ".." in relative.parts or name in files or name == manifest.name):
            raise ValueError(f"invalid transfer manifest entry: {name}")
        files[name] = checksum
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != files.keys() | {manifest.name}:
        raise ValueError("received file coverage differs from the transmission manifest")
    for name, checksum in files.items():
        if sha256(root / name) != checksum:
            raise ValueError(f"received checksum mismatch: {name}")
    return {"payloads": len(files), "bytes": sum((root / f).stat().st_size for f in files),
            "manifest_sha256": sha256(manifest)}


def remote_command(kind: str, arms: list[str], methods: list[str], wait_seconds: int) -> str:
    common = f"""set -euo pipefail
cd '{REMOTE_RUN}'
deadline=$(( $(date +%s) + {wait_seconds} ))
"""
    if kind == "external":
        body = f"""
files=()
for m in {' '.join(methods)}; do
    control="control/external"
    exit_file="$control/$m.actual_exit"
    if [[ "$m" == "mmcad_v2" && -d control/mmcad_v2_retry1 ]]; then
        control="control/mmcad_v2_retry1"
        exit_file="$control/actual_exit"
    fi
    until [[ -s "$exit_file" ]]; do
        if (( $(date +%s) >= deadline )); then printf 'Waiting for %s\n' "$m" >&2; exit 4; fi
        sleep 15
    done
    [[ "$(cat "$exit_file")" == 0 ]] || {{ printf '%s failed\n' "$m" >&2; exit 3; }}
    files+=("$exit_file" "external/$m/l0_replay.json")
    for stem in gallery {' '.join("texts_" + arm for arm in arms)}; do
        files+=("external/$m/$stem.npy" "external/$m/${{stem}}_status.jsonl" "external/$m/${{stem}}_meta.json")
    done
    if [[ "$control" == "control/external" ]]; then
        files+=("$control/$m.stdout.log" "$control/$m.stderr.log" "$control/$m.argv" "$control/$m.pid")
    else
        files+=("$control/stdout.log" "$control/stderr.log" "$control/launch_command.txt" "$control/launcher.pid")
        files+=("control/external/$m.actual_exit" "control/external/$m.stderr.log")
    fi
done
"""
    else:
        body = f"""
files=(full/run_authorization.json full/gallery.jsonl full/capacity_projection.json)
for arm in {' '.join(arms)}; do
    marker="full/${{arm}}_complete.json"
    until [[ -s "$marker" ]]; do
        if [[ -s control/score/controller.actual_exit ]]; then
            printf 'Controller exited without the complete %s artifact\n' "$arm" >&2
            exit 3
        fi
        if (( $(date +%s) >= deadline )); then printf 'Waiting for %s\n' "$arm" >&2; exit 4; fi
        sleep 15
    done
    files+=("$marker" "full/scores/$arm/text_tokens.json")
    files+=(full/scores/"$arm"/t[0-9]*.npy full/scores/"$arm"/t[0-9]*.json)
    files+=(full/"$arm"_worker*.json full/"$arm"_exit_*.json full/"$arm"_shard*.launch.json)
    files+=(full/"$arm"_shard*.stdout.log full/"$arm"_shard*.stderr.log)
done
"""
        if "l2" in arms:
            body += """
until [[ -s control/score/controller.actual_exit ]]; do
    if (( $(date +%s) >= deadline )); then printf 'Waiting for final scorer exit\n' >&2; exit 4; fi
    sleep 5
done
[[ "$(cat control/score/controller.actual_exit)" == 0 ]]
"""
        body += """
if [[ -s control/score/controller.actual_exit ]]; then
    files+=(control/score/controller.actual_exit full/ledger.json control/score/launch_command.txt)
    if [[ -f full/controller_exit.json ]]; then files+=(full/controller_exit.json); fi
fi
"""
    # The target has ordinary GNU tar and sha256sum. This temporary file is a
    # transmission receipt; the model artifacts are only read.
    return common + body + r"""
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/TRANSFER_SHA256SUMS"; rmdir -- "$scratch"' EXIT
sha256sum -- "${files[@]}" > "$scratch/TRANSFER_SHA256SUMS"
tar -cf - "${files[@]}" -C "$scratch" TRANSFER_SHA256SUMS
"""


def collect(args) -> int:
    args.output.mkdir()
    archive = args.output / "artifact.tar"
    command = remote_command(args.kind, args.condition, args.method, args.wait_seconds)
    with archive.open("wb") as payload, (args.output / "ssh.stderr.log").open("wb") as err:
        with subprocess.Popen(
            ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
             "-o", "ServerAliveCountMax=2", "yunnao-share4", "tr -d '\\r' | bash -s"],
            stdin=subprocess.PIPE, stdout=payload, stderr=err,
        ) as result:
            started = time.time()
            start_file = args.output / "connection_start.json"
            tmp = start_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"ssh_pid": result.pid, "started_epoch": started,
                                       "kind": args.kind, "conditions": args.condition}) + "\n",
                           encoding="utf-8")
            tmp.replace(start_file)
            result.communicate(input=command.encode("ascii"))
    record = {"actual_ssh_exit": result.returncode, "started_epoch": started, "ended_epoch": time.time(),
              "kind": args.kind, "conditions": args.condition, "methods": args.method,
              "collector_sha256": sha256(Path(__file__))}
    if result.returncode == 0:
        received = args.output / "raw"
        received.mkdir()
        with tarfile.open(archive, "r:") as stream:
            stream.extractall(received, filter="data")
        record.update(verify_received(received))
    receipt = args.output / "pull_receipt.json"
    staged = receipt.with_suffix(".tmp")
    staged.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    staged.replace(receipt)
    print(json.dumps(record), flush=True)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("external", "text2cad"))
    parser.add_argument("--condition", action="append", choices=CONDITIONS, required=True)
    parser.add_argument("--method", action="append", choices=CORE_EXTERNAL)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.kind == "external" and args.method is None:
        args.method = list(CORE_EXTERNAL)
    if args.kind == "text2cad" and args.method:
        parser.error("--method applies only to the external methods")
    if len(args.condition) != len(set(args.condition)) or (args.method and len(args.method) != len(set(args.method))):
        parser.error("refusing repeated collection selections")
    if args.wait_seconds < 0:
        parser.error("--wait-seconds is nonnegative")
    raise SystemExit(collect(args))


if __name__ == "__main__":
    main()
