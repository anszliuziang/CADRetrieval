"""E2 empty-text background scorer: dedicated single-card entry (plan
IDEA01_E2_EMPTY_TEXT_BACKGROUND_2026-11-12.md, v2).

Three frozen background manifests (built by build_text2cad_e2_manifests.py,
fingerprints pinned in the committed configs/text2cad_e2_background_expected_20261112.json):

  b0      texts = [""]               -> 1 x 8046 pairs (primary background)
  b1      texts = ["a CAD model."]   -> 1 x 8046 pairs (pre-registered sensitivity)
  replay  texts = 8 pinned G4 texts  -> 8 x 1024 pairs (cross-machine comparability
          gate against share2 G4 records BEFORE any differencing; tolerance
          |ds| <= max(1e-3, 1e-6*|s|) and n_valid equality, checked analysis-side)

Gates (identical functions for --mode dry-run and --mode score, per review):
  * committed expected-config fingerprints for all three manifests;
  * B0/B1 admitted ONLY via idea01.text2cad_pair_index.load_background_texts
    (the general load_texts contract stays untouched);
  * candidate store byte-identical to the original pair_manifest_v4 cad_store
    (b0/b1: full 8,046; replay: first 1,024 rows, a prefix so cad_index values
    equal the G4 cad_index);
  * tokenizer evidence for B0/B1 recorded ([CLS]/[SEP] expectation for B0),
    refuse >512 BERT tokens;
  * score mode ONLY: ACTIVE_TASK fetched from the plan-pinned authorization ref
    on an allowlisted origin (CNB / GitHub mirror) must state pilot_authorized,
    e2_approved_sha == detached clean HEAD, positive e2_max_pairs /
    e2_max_card_seconds and a parseable e2_deadline_utc; the E2 budget ledger
    enforces the pair/card-second caps and the wall-clock deadline across all
    invocations (smoke + batches + replay all share the ledger).

Score mode reuses the reviewed G4 machinery (model load gates, FrozenMatrixScorer,
SharedBudgetLedger) via lazy imports; this module stays torch-free at import so
CPU dry-run and unit tests run without torch. Protocol identical to G4:
float32, full 272 positions, no trim/cache, batch 16 (honest tail).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idea01.p1_runner import (  # noqa: E402
    active_task_field,
    fetch_authorized_active_task,
)
from idea01.text2cad_pair_index import (  # noqa: E402
    CAD_STORE_FILE,
    PAIR_MANIFEST_FILE,
    REASON_BACKGROUND_TEXT_MISMATCH,
    TEXTS_FILE,
    load_background_texts,
    load_cad_store,
    load_pair_manifest,
    load_texts,
    sha256_file,
)

E2_CONFIG_REL = "experiments/configs/text2cad_e2_background_expected_20261112.json"
B0_TEXT = ""
B1_TEXT = "a CAD model."
BACKGROUND_TEXTS = {"b0": B0_TEXT, "b1": B1_TEXT}
BATCH_SIZE = 16  # G4 protocol (honest tail)
REQUIRED_COUNTS = {"b0": (1, 8046), "b1": (1, 8046), "replay": (8, 1024)}


class E2Error(RuntimeError):
    """Hard refusal; exit non-zero, no forward."""


def load_e2_config(repo_root: Path) -> dict:
    path = Path(repo_root) / E2_CONFIG_REL
    if not path.is_file():
        raise E2Error(f"committed E2 expected-config missing: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema", "authorization_ref", "backgrounds", "original_cad_store_sha256",
                "replay_texts", "cad_seq_zip_sha256"):
        if key not in cfg:
            raise E2Error(f"E2 expected-config missing key {key!r}")
    if cfg["schema"] != "text2cad-e2-background-expected-v1":
        raise E2Error(f"E2 config schema mismatch: {cfg['schema']!r}")
    return cfg


def git_head(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(repo_root), capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def git_dirty(repo_root: Path) -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=Path(repo_root), capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _manifest_fingerprint_ok(cfg: dict, which: str, manifest_dir: Path) -> dict:
    pins = cfg["backgrounds"][which]
    manifest = load_pair_manifest(
        manifest_dir,
        expected_sha256={
            TEXTS_FILE: pins["texts_sha256"],
            CAD_STORE_FILE: pins["cad_store_sha256"],
            "cad_length_index.jsonl": pins["cad_length_index_sha256"],
            PAIR_MANIFEST_FILE: pins["pair_manifest_sha256"],
        },
    )
    counts = manifest["counts"]
    want_t, want_c = REQUIRED_COUNTS[which]
    if counts.get("n_unique_text") != want_t or counts.get("n_gallery") != want_c:
        raise E2Error(
            f"{which}: counts {counts.get('n_unique_text')}x{counts.get('n_gallery')} != {want_t}x{want_c}"
        )
    return manifest


def validate_e2_manifests(cfg: dict, dirs: dict[str, Path], original_manifest_dir: Path) -> dict:
    """Shared validation for dry-run AND score (identical checks, per review)."""
    report: dict[str, Any] = {"backgrounds": {}}
    original_store_sha = sha256_file(Path(original_manifest_dir) / CAD_STORE_FILE)
    if original_store_sha != cfg["original_cad_store_sha256"]:
        raise E2Error(
            f"original pair_manifest_v4 cad_store sha {original_store_sha} != pinned "
            f"{cfg['original_cad_store_sha256']}"
        )
    for which in ("b0", "b1"):
        d = Path(dirs[which])
        _manifest_fingerprint_ok(cfg, which, d)
        text = load_background_texts(d, BACKGROUND_TEXTS[which])
        store_sha = sha256_file(d / CAD_STORE_FILE)
        if store_sha != cfg["original_cad_store_sha256"]:
            raise E2Error(f"{which}: cad_store must be byte-identical to original v4, sha {store_sha}")
        members = load_cad_store(d)
        if len(members) != 8046:
            raise E2Error(f"{which}: cad store has {len(members)} members, expected 8046")
        report["backgrounds"][which] = {"text": text, "n_cads": len(members)}
    d = Path(dirs["replay"])
    _manifest_fingerprint_ok(cfg, "replay", d)
    texts = load_texts(d)
    pinned = cfg["replay_texts"]
    if len(texts) != len(pinned):
        raise E2Error(f"replay texts {len(texts)} != pinned {len(pinned)}")
    for i, (got, want) in enumerate(zip(texts, pinned)):
        if got != want["text"]:
            raise E2Error(f"replay text {i} differs from pinned original text {want['original_text_index']}")
    members = load_cad_store(d)
    if len(members) != 1024:
        raise E2Error(f"replay cad store has {len(members)} members, expected 1024 (prefix)")
    report["backgrounds"]["replay"] = {"n_texts": len(texts), "n_cads": len(members)}
    return report


def tokenizer_evidence(bert_dir: Path) -> dict:
    """Slow BERT tokenizer evidence for B0/B1 (dry-run + score share this)."""
    from transformers import BertTokenizer  # tokenizer-only; no torch needed

    tok = BertTokenizer.from_pretrained(str(bert_dir), use_fast=False)
    ev: dict[str, Any] = {}
    for name, text in BACKGROUND_TEXTS.items():
        ids = tok(text, add_special_tokens=True)["input_ids"]
        if not ids or len(ids) > 512:
            raise E2Error(f"{name}: background encoding empty or >512 tokens: {ids!r}")
        ev[name] = {"text": text, "input_ids": ids, "n_tokens": len(ids)}
    ev["b0_expectation"] = "CLS/SEP-only (ids [101, 102] for bert-large-uncased vocab); recorded, not hardcoded as a gate"
    return ev


def parse_e2_authorization(text: str, approved_sha: str) -> dict:
    state = active_task_field(text, "state")
    sha = active_task_field(text, "e2_approved_sha")
    max_pairs = active_task_field(text, "e2_max_pairs")
    max_card_seconds = active_task_field(text, "e2_max_card_seconds")
    deadline = active_task_field(text, "e2_deadline_utc")
    problems: list[str] = []
    if state != "pilot_authorized":
        problems.append(f"state is {state!r}, not pilot_authorized")
    if not sha or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        problems.append(f"e2_approved_sha missing/malformed: {sha!r}")
    elif sha != approved_sha:
        problems.append(f"e2_approved_sha {sha} != HEAD {approved_sha}")
    caps: dict[str, Any] = {}
    try:
        caps["max_pairs"] = int(max_pairs or "")
        if caps["max_pairs"] <= 0:
            raise ValueError
    except ValueError:
        problems.append(f"e2_max_pairs missing/non-positive: {max_pairs!r}")
    try:
        caps["max_card_seconds"] = float(max_card_seconds or "")
        if caps["max_card_seconds"] <= 0:
            raise ValueError
    except ValueError:
        problems.append(f"e2_max_card_seconds missing/non-positive: {max_card_seconds!r}")
    try:
        dt = datetime.strptime(deadline or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        caps["deadline_epoch"] = dt.timestamp()
    except ValueError:
        problems.append(f"e2_deadline_utc missing/unparseable (want YYYY-MM-DDTHH:MM:SSZ): {deadline!r}")
    if problems:
        raise E2Error("E2 authorization refused: " + "; ".join(problems))
    return {"approved_sha": sha, **caps}


def authorize_e2_run(repo_root: Path, cfg: dict) -> dict:
    ref = cfg["authorization_ref"]
    text, meta = fetch_authorized_active_task(Path(repo_root), ref=ref)  # allowlisted origin
    if text is None:
        raise E2Error(f"authorization fetch refused: {meta.get('problems')}")
    head = git_head(repo_root)
    auth = parse_e2_authorization(text, head)
    auth["authorization_meta"] = meta
    auth["authorization_ref"] = ref
    return auth


def check_deadline(auth: dict, now: Optional[float] = None) -> None:
    if (now if now is not None else time.time()) > auth["deadline_epoch"]:
        raise E2Error(
            f"e2_deadline_utc passed ({auth['deadline_epoch']}); no further forwards allowed"
        )


def dry_run_e2(cfg: dict, dirs: dict[str, Path], original_manifest_dir: Path, bert_dir: Path,
               cad_seq_zip: Path, out_dir: Path) -> dict:
    """Forward-free CPU preflight: identical input validation + tokenizer evidence."""
    report = validate_e2_manifests(cfg, dirs, original_manifest_dir)
    zip_sha = sha256_file(cad_seq_zip)
    if zip_sha != cfg["cad_seq_zip_sha256"]:
        raise E2Error(f"cad_seq.zip sha {zip_sha} != pinned {cfg['cad_seq_zip_sha256']}")
    report["cad_seq_zip_sha256"] = zip_sha
    report["tokenizer_evidence"] = tokenizer_evidence(Path(bert_dir))
    report["status"] = "dry_run_no_forward"
    report["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_path = Path(out_dir) / "dry_run.json"
    if out_path.exists():
        raise E2Error(f"refusing to overwrite {out_path}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def score_e2(repo_root: Path, cfg: dict, which: str, dirs: dict[str, Path],
             checkpoint: Path, bert_dir: Path, cad_seq_zip: Path,
             out_dir: Path, device: str) -> int:
    """Single-card scoring of one background manifest; reuses the reviewed G4 machinery."""
    repo_root = Path(repo_root)
    # lazy import: torch-bearing modules only on the score path
    sys.path.insert(0, str(repo_root / "experiments" / "src"))
    from idea01.text2cad_frozen_runner import (  # noqa: E402
        CHUNK_SCHEMA_VERSION,
        PINNED_CHECKPOINT_SHA256,
        SCORE_NAME,
        SCORE_SPEC_VERSION,
        FrozenMatrixScorer,
        RunnerConfig,
        SharedBudgetLedger,
        check_revision_paths,
        chunk_file_name,
        ensure_outside_checkout,
        load_gated_model_and_store,
        plan_chunks,
    )
    from idea01.text2cad_pair_index import PairSpace  # noqa: E402

    if which not in BACKGROUND_TEXTS:
        raise E2Error(f"--which must be one of {sorted(BACKGROUND_TEXTS)}, got {which!r}")
    if git_dirty(repo_root):
        raise E2Error(f"git worktree not clean:\n{git_dirty(repo_root)}")

    auth = authorize_e2_run(repo_root, cfg)
    check_deadline(auth)
    validate_e2_manifests(cfg, dirs, dirs["_original"])  # identical checks as dry-run

    manifest_dir = Path(dirs[which])
    pins = cfg["backgrounds"][which]
    n_t, n_c = REQUIRED_COUNTS[which]
    pair_space = PairSpace(n_texts=n_t, n_cads=n_c)
    if which == "replay":
        texts = load_texts(manifest_dir)
    else:
        texts = [load_background_texts(manifest_dir, BACKGROUND_TEXTS[which])]
    members = load_cad_store(manifest_dir)

    cfg_run = RunnerConfig(
        checkpoint_path=str(checkpoint), bert_dir=str(bert_dir), cad_seq_zip=str(cad_seq_zip),
        pair_manifest_dir=str(manifest_dir), device=device, torch_dtype="float32",
        batch_size=BATCH_SIZE, expected_keys_manifest=str(repo_root / "experiments" / "configs" / "text2cad_checkpoint_expected_keys_20260930.json"),
        trim_mode="none", text_cache=False, shard_id=0, n_shards=1, texts_per_chunk=1,
    )
    check_revision_paths(cfg_run)
    run_root = Path(out_dir).resolve()
    ensure_outside_checkout(run_root, repo_root, "out_dir")
    ledger_path = run_root / "e2_budget_ledger.json"
    ledger = SharedBudgetLedger(
        ledger_path, approved_sha=auth["approved_sha"], max_pairs=auth["max_pairs"],
        max_card_seconds=auth["max_card_seconds"],
    )
    planned = n_t * n_c
    if planned > ledger.remaining_pairs():
        raise E2Error(f"planned {planned} pairs exceed remaining e2 budget {ledger.remaining_pairs()}")
    check_deadline(auth)

    fingerprints = {PAIR_MANIFEST_FILE: pins["pair_manifest_sha256"]}
    binding = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "pair_manifest_sha256": pins["pair_manifest_sha256"],
        "git_sha": auth["approved_sha"],
        "checkpoint_sha256": PINNED_CHECKPOINT_SHA256,
        "device": device,
        "torch_dtype": "float32",
        "trim_mode": "none",
        "text_cache": False,
        "score_name": SCORE_NAME,
        "spec_version": SCORE_SPEC_VERSION,
        "e2_background": which,
        "authorization_ref": auth["authorization_ref"],
    }
    todo, completed = plan_chunks(run_root, pair_space, cfg_run, binding, resume=False)
    if len(todo) != n_t:
        raise E2Error(f"plan has {len(todo)} chunks, expected {n_t}")

    invocation_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ledger.begin_invocation(invocation_id, shard_id=0, cards_used=1)
    exit_code, note, failure, published = 0, "ok", None, 0
    try:
        model, load_report, store, preload_report = load_gated_model_and_store(cfg_run, members)
        scorer = FrozenMatrixScorer(cfg_run, model, texts, store, pair_space, ledger, invocation_id, binding)
        for t0, t1, _p in todo:
            check_deadline(auth)
            chunk_records = scorer.score_chunk(t0, t1)
            payload = [json.dumps(r, ensure_ascii=False) for r in chunk_records]
            target = run_root / chunk_file_name(0, t0, t1)
            tmp = target.with_suffix(".tmp")
            tmp.write_text("\n".join(payload) + "\n", encoding="utf-8", newline="\n")
            tmp.replace(target)
            published += 1
    except Exception as exc:  # noqa: BLE001
        exit_code, note, failure = 2, "aborted", f"{type(exc).__name__}: {exc}"
    finally:
        ledger.finalize_invocation(invocation_id, exit_code=exit_code, note=note)
    summary = {
        "invocation_id": invocation_id, "e2_background": which, "authorization_ref": auth["authorization_ref"],
        "approved_sha": auth["approved_sha"], "device": device, "published": published,
        "planned_pairs": planned, "ledger": ledger.to_dict(), "failure": failure,
        "authorization_meta": auth["authorization_meta"],
    }
    (run_root / f"summary_{invocation_id}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if failure:
        print(f"E2 FATAL: {failure}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "invocation_id": invocation_id, "background": which,
                      "pairs": planned, "ledger_pairs": ledger.pairs_charged()}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["dry-run", "score"], default="dry-run")
    ap.add_argument("--checkpoint")
    ap.add_argument("--bert-dir", required=True)
    ap.add_argument("--cad-seq-zip")
    ap.add_argument("--b0-manifest-dir", required=True)
    ap.add_argument("--b1-manifest-dir", required=True)
    ap.add_argument("--replay-manifest-dir", required=True)
    ap.add_argument("--original-manifest-dir", required=True,
                    help="original pair_manifest_v4 dir (cad_store equality reference)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--which", choices=["b0", "b1", "replay"], default=None,
                    help="score mode: which background manifest to score")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args(argv)
    try:
        repo_root = Path(args.repo_root).resolve()
        cfg = load_e2_config(repo_root)
        dirs = {"b0": Path(args.b0_manifest_dir), "b1": Path(args.b1_manifest_dir),
                "replay": Path(args.replay_manifest_dir), "_original": Path(args.original_manifest_dir)}
        if args.mode == "dry-run":
            if not args.cad_seq_zip:
                raise E2Error("--cad-seq-zip required for dry-run")
            report = dry_run_e2(cfg, dirs, Path(args.original_manifest_dir), Path(args.bert_dir),
                                Path(args.cad_seq_zip), Path(args.out_dir))
            print(json.dumps({"status": report["status"], "tokenizer": report["tokenizer_evidence"]}))
            return 0
        # score mode
        for req in (args.checkpoint, args.cad_seq_zip, args.which):
            if not req:
                raise E2Error("score mode requires --checkpoint, --cad-seq-zip and --which")
        return score_e2(repo_root, cfg, args.which, dirs, Path(args.checkpoint), Path(args.bert_dir),
                        Path(args.cad_seq_zip), Path(args.out_dir), args.device)
    except Exception as exc:  # noqa: BLE001 — every refusal exits non-zero with the reason
        print(f"E2 FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
