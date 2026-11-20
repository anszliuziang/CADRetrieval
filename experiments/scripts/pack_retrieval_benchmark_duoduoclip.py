"""Transfer only the verified eight-view data and reviewed Git bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import tarfile
import time

from encode_retrieval_benchmark_duoduoclip import SPEC, load_render_axis, scoring_inputs, write_receipt
from prepare_retrieval_benchmark_fill import load_json, sha256


RENDER_CODE = "8f3172af76b04fb7daab6fd46a60a81a11aebe05"


def add_bytes(archive, member, data):
    info = tarfile.TarInfo(member)
    info.size = len(data)
    info.mode = 0o640
    archive.addfile(info, io.BytesIO(data))


def pack(run: Path, bundle: Path, output: Path) -> dict:
    exit_path = run / "control/duoduo_full_render_exit.json"
    exit_record = load_json(exit_path)
    if exit_record.get("actual_exit") != 0 or exit_record.get("implementation_sha") != RENDER_CODE:
        raise ValueError("missing a successful run of the approved eight-view renderer")
    render = run / "duoduoclip_render"
    spec = load_json(SPEC)
    binding = spec["deepcad"]
    gallery, _, _ = scoring_inputs(run / "frozen", binding, binding["conditions"])
    records, meta = load_render_axis(render, gallery, binding, sha256(SPEC))
    output.mkdir()
    payload = output / "payload.tar"
    entries = {}
    total = 0
    with tarfile.open(payload, "x") as archive:
        for source, member in (
            (bundle, "work.bundle"),
            (exit_path, "render.actual_exit.json"),
            (render / "render_summary.json", "renders/render_summary.json"),
            (render / "gallery_render_status.jsonl", "renders/gallery_render_status.jsonl"),
        ):
            data = source.read_bytes()
            entries[member] = hashlib.sha256(data).hexdigest()
            total += len(data)
            add_bytes(archive, member, data)
        for record in records:
            if not record["valid"]:
                continue
            folder = record["uid"].replace("/", "__")
            for view in record["views"]:
                image = Path("views") / folder / view["filename"]
                path = render / image
                if path.is_symlink():
                    raise ValueError(f"Unexpected symbolic link: {path}")
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if len(data) != view["bytes"] or digest != view["sha256"]:
                    raise ValueError(f"Rendered image changed after completion: {path}")
                member = "renders/" + image.as_posix()
                entries[member] = digest
                total += len(data)
                add_bytes(archive, member, data)
        if (len(entries) != 4 + 8 * sum(row["valid"] for row in records)
                or entries["renders/render_summary.json"] != meta["render_summary_sha256"]
                or entries["renders/gallery_render_status.jsonl"] != meta["render_status_sha256"]):
            raise ValueError("Transfer coverage differs from the approved render summary")
        manifest = "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())).encode("ascii")
        add_bytes(archive, "SHA256SUMS", manifest)
    (output / "SHA256SUMS").write_bytes(manifest)
    return {"files": len(entries), "bytes": total, "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "tar_sha256": sha256(payload), "bundle_sha256": entries["work.bundle"],
            "valid_renders": sum(row["valid"] for row in records), "all_renders": len(records), **meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--code-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    stop = time.time() + args.wait_seconds
    exit_path = args.runtime / "control/duoduo_full_render_exit.json"
    while not exit_path.is_file() and time.time() < stop:
        time.sleep(15)
    receipt = pack(args.runtime, args.code_bundle, args.output)
    write_receipt(args.output / "pack_receipt.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
