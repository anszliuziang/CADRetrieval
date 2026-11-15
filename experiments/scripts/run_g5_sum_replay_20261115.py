"""SUM replay through the unified G5 evaluation entry (plan
IDEA01_G5_EXTERNAL_BASELINES_2026-11-15.md section 2, deliverable 1+2).

Recomputes SUM ranks for all 3,644 x 8,046 certified G4 pairs, exports them
as an EXTERNAL score artifact (npy matrix + identity sidecar), reloads the
artifact through text2cad_g5_eval_entry.evaluate (the same entry external
methods will use), and reconciles every query against the historical G5 run
(query_id, positive cad_id, SUM rank — 7,934/7,934 required) plus the
historical metrics (hit counts 16/73/130/207, MedR 3411.5).

Zero model forwards; local CPU only; new independent output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import text2cad_g5_cpu as g5  # noqa: E402
from text2cad_g5_eval_entry import ScoreSet, evaluate, load_frozen_protocol  # noqa: E402

G4_RUN = Path("experiments/runs/idea01_text2cad_g4_full_20261002")
G5_RUN = Path("experiments/runs/idea01_text2cad_g5_cpu_20261003")
EXPECTED_HITS = {1: 16, 5: 73, 10: 130, 20: 207}
EXPECTED_MEDR = 3411.5
EXPECTED_CUTS = [27.0, 40.0, 70.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--runtime-root", default="D:/projects/IdeaSpark/CADRetrievalIdea1_runtime")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    repo, output = Path(a.repo), Path(a.output)
    started = time.time()

    protocol = load_frozen_protocol(repo, Path(a.data_root), Path(a.runtime_root),
                                    require_certificate=True)
    linked = protocol["linked"]
    lengths = linked["lengths"]
    config = protocol["config"]

    entries = g5.manifest_entries(repo / G4_RUN / "SHA256SUMS")
    g5.require(g5.sha256(repo / G4_RUN / "final_ledger.json") ==
               g5.read_json(repo / config["certificate_path"])["final_ledger_sha256"],
               "final score ledger differs")
    g5.require(a.output and not Path(a.output).exists(), "existing output directory")

    totals = np.empty((3644, 8046), dtype=np.float64)
    for t in range(3644):
        shard = t // 1822
        name = f"card{shard}/chunk_s{shard:02d}_t{t:05d}_{t + 1:05d}.jsonl"
        total, _mean, _n = g5.load_chunk(repo / G4_RUN / name, entries[name], t, 8046,
                                         config["inputs"]["pair_manifest"]["sha256"],
                                         config["checkpoint_sha256"], lengths)
        totals[t] = total
        if (t + 1) % 400 == 0 or t == 3643:
            print(json.dumps(dict(phase="matrix_read", completed_texts=t + 1,
                                  seconds=round(time.time() - started, 1))), flush=True)

    # export the SUM scores as an external artifact and reload it through the
    # entry — the evaluator must consume the artifact, not the reader's memory
    output = Path(a.output)
    output.mkdir(parents=True)
    npy = output / "sum_scores_full_axis_f64.npy"
    npy.write_bytes(np.ascontiguousarray(totals, dtype="<f8").tobytes())
    sidecar = dict(schema="text2cad-method-scores-sidecar-v1", method="text2cad_sum_replay",
                   direction="higher", dtype="float64",
                   text_indices=list(range(3644)), cad_indices=list(range(8046)),
                   matrix_sha256=hashlib.sha256(npy.read_bytes()).hexdigest(),
                   matrix_bytes=npy.stat().st_size,
                   source_manifest_sha256=g5.MANIFEST_SHA,
                   source_certificate_sha256=g5.CERTIFICATE_SHA,
                   zero_npu=True, score_forwards=0)
    side = output / "sum_scores_sidecar.json"
    side.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8", newline="\n")
    reloaded = np.fromfile(npy, dtype="<f8").reshape(3644, 8046)
    g5.require(hashlib.sha256(npy.read_bytes()).hexdigest() == sidecar["matrix_sha256"],
               "external score artifact SHA differs after reload")
    score_set = ScoreSet.from_matrix(
        reloaded, sidecar["text_indices"], sidecar["cad_indices"],
        direction=sidecar["direction"], dtype=sidecar["dtype"],
        meta=dict(name="text2cad_sum_replay", sidecar=str(side),
                  sidecar_sha256=hashlib.sha256(side.read_bytes()).hexdigest(),
                  matrix_sha256=sidecar["matrix_sha256"],
                  source_manifest_sha256=g5.MANIFEST_SHA,
                  source_certificate_sha256=g5.CERTIFICATE_SHA))

    eval_out = output / "entry_evaluation"
    results = evaluate(score_set, protocol=protocol, output=eval_out,
                       method_meta=dict(name="text2cad_sum_replay",
                                        score="two_head_factorized_logprob_sum",
                                        spec_version="text2cad-vector-score-v1",
                                        source="G4 certified matrix replay"),
                       cad_subset=None, paired_sum_set=None, bootstrap=None)

    # ---- per-query reconciliation against the historical G5 run ----
    hist = {}
    with (repo / G5_RUN / "query_ranks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            hist[row["query_id"]] = row
    recon_rows = []
    n_match = 0
    with (eval_out / "query_ranks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["query_id"]
            h = hist[qid]
            ok = (row["cad_id"] == h["cad_id"] and row["text_index"] == h["text_index"]
                  and row["ranks"]["text2cad_sum_replay"] == h["ranks"]["sum"])
            n_match += int(ok)
            recon_rows.append(dict(query_id=qid, match=bool(ok)))
    g5.require(n_match == 7934 and len(recon_rows) == 7934,
               f"historical reconciliation failed: {n_match}/7934")

    hist_metrics = g5.read_json(repo / G5_RUN / "metrics.json")["metrics"]["sum"]
    method = "text2cad_sum_replay"
    hit_ok = True
    for k, want in EXPECTED_HITS.items():
        got = results["hit_counts"][method][f"hit@{k}"]
        hit_ok &= (got == want)
        g5.require(got == want, f"hit@{k} {got} != expected {want}")
        g5.require(results["metrics"][method][f"R@{k}"] == hist_metrics[f"R@{k}"],
                   f"R@{k} fraction differs from historical metrics.json")
    g5.require(results["metrics"][method]["MedR"] == hist_metrics["MedR"] == EXPECTED_MEDR,
               "MedR differs from historical/expected")
    g5.require(results["stratification"]["gallery_length_quartile_cutpoints"] == EXPECTED_CUTS,
               "gallery length quartile cutpoints differ from the frozen protocol")

    recon = dict(schema="text2cad-g5-sum-replay-reconciliation-v1",
                 per_query_match=int(n_match), per_query_total=len(recon_rows),
                 identity_triple=("query_id", "positive_cad_id", "sum_rank"),
                 historical_query_ranks=str(repo / G5_RUN / "query_ranks.jsonl"),
                 historical_metrics=str(repo / G5_RUN / "metrics.json"),
                 hit_counts={f"hit@{k}": results["hit_counts"][method][f"hit@{k}"]
                             for k in EXPECTED_HITS},
                 hit_counts_expected=EXPECTED_HITS,
                 displayed={f"R@{k}": results["metrics"][method][f"R@{k}"] for k in EXPECTED_HITS},
                 MedR=results["metrics"][method]["MedR"],
                 quartile_cutpoints=results["stratification"]["gallery_length_quartile_cutpoints"],
                 all_equal=True)
    (output / "reconciliation.json").write_text(
        json.dumps(recon, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")

    lines = []
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.relative_to(output).as_posix())
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status="complete", output=str(output),
                          per_query_match=int(n_match), per_query_total=7934,
                          hit_counts=recon["hit_counts"], MedR=recon["MedR"],
                          elapsed_seconds=round(time.time() - started, 1),
                          zero_npu=True, score_forwards=0)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
