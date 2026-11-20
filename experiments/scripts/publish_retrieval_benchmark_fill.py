"""Publish verified DeepCAD retrieval cells into the three benchmark report files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

from eval_retrieval_benchmark_fill import (
    CORE_METHODS,
    K_LEVELS,
    FrozenCondition,
    read_rank_results,
    summarize,
)
from prepare_retrieval_benchmark_fill import ASSETS, CONDITIONS, COUNTS, load_json, sha256, verify_sha256


MARKDOWN = "RETRIEVAL_COMPARISON_TABLES.md"
TEX = "RETRIEVAL_COMPARISON_TABLES.tex"
EVIDENCE = "RETRIEVAL_TABLE_EVIDENCE.json"
REPORT_FILES = (MARKDOWN, TEX, EVIDENCE)
ROWS = {
    "clip": ("CLIP ViT-B/32，单视图", "CLIP ViT-B/32 (one view)"),
    "openshape": ("OpenShape，点云适配版", "OpenShape (point cloud)"),
    "mmcad_v2": ("MM-CAD，修正版 v2", "MM-CAD (v2 pipeline)"),
    "duoduoclip": ("DuoduoCLIP-B/32，八视图", "DuoduoCLIP-B/32 (8 views)"),
    "sequence_dual_encoder": ("文本–命令序列双塔", "Text--sequence dual encoder"),
    "spectrabrep": ("SpectraBRep，0.6B LoRA", "SpectraBRep (0.6B LoRA)"),
    "imma": ("IMMA，序列＋点云¹", "IMMA (sequence + points)$^{a}$"),
    "generate_openshape": ("Text2CAD 生成后用 OpenShape 匹配", "Generate then match (OpenShape)"),
    "sum": ("Text2CAD 原生分 S₀", "Text2CAD raw score"),
    "s48": ("Text2CAD 背景校正 S₄₈", "Text2CAD background-corrected"),
    "spectrabrep_ct": ("SpectraBRep，短句监督重训²", "SpectraBRep (short-caption retrain)$^{b}$"),
}
ARM_LABELS = {"l1": "L1", "l2": "L2", "l3": "L3", "ct_deepcad": "CadTranslator–DeepCAD"}
STATUS_MARKER = "retrieval-benchmark-fill"
PENDING_CT = "表 2 尚无符合此清单的模型实测行。"
TEX_PENDING = r"\multicolumn{5}{c}{---}"


def audit_rollup(rows: list[dict], metrics: dict, gallery_size: int, label: str) -> None:
    """Verify hit integers, percents and MedR independently from the evaluator's aggregation."""
    ranked = [row["rank"] for row in rows]
    n = len(ranked)
    hits = [sum(rank <= min(k, gallery_size) for rank in ranked) for k in K_LEVELS]
    percents = [100 * hit / n if n else None for hit in hits]
    medr = statistics.median(ranked) if n else None
    if (metrics["queries"] != n or metrics["hits"] != hits or metrics["median_rank"] != medr
            or len(metrics["recall_percent"]) != len(percents)
            or any((actual is not None if expected is None else
                    actual is None or not math.isclose(actual, expected, rel_tol=1e-14))
                   for actual, expected in zip(metrics["recall_percent"], percents, strict=True))):
        raise ValueError(f"Published rollup does not match its verified query ranks: {label}")


def audit_source(metrics: dict, condition: str) -> None:
    source = metrics["source"]
    path = Path(source["path"])
    text = f"texts_{condition}"
    mode = source["mode"]
    if mode == "embeddings":
        pins = {
            path / "gallery.npy": source["gallery_embeddings_sha256"],
            path / f"{text}.npy": source["text_embeddings_sha256"],
            path / "gallery_meta.json": source["gallery_meta_sha256"],
            path / f"{text}_meta.json": source["text_meta_sha256"],
        }
        if "l0_replay_sha256" in source:
            pins[path / "l0_replay.json"] = source["l0_replay_sha256"]
    elif mode == "scores":
        pins = {
            path / f"scores_{condition}.npy": source["matrix_sha256"],
            path / f"scores_{condition}_meta.json": source["meta_sha256"],
        }
    elif mode == "Text2CAD authorized per-text raw sum shards":
        run = path.parents[1]
        pins = {
            run / "run_authorization.json": source["authorization_sha256"],
            run / f"{condition}_complete.json": source["completion_sha256"],
            path / "text_tokens.json": source["token_metadata_sha256"],
        }
    else:
        raise ValueError(f"Unrecognized evidence source for {condition}/{metrics['method']}: {mode}")
    if mode in ("embeddings", "scores"):
        pins[path / "gallery_status.jsonl"] = source["gallery_status_sha256"]
        pins[path / f"{text}_status.jsonl"] = source["text_status_sha256"]
    for artifact, digest in pins.items():
        verify_sha256(artifact, digest)


def audit_results(runtime: Path) -> tuple[dict, dict]:
    frozen_root = runtime / "frozen"
    evaluation = runtime / "evaluation"
    history_path = evaluation / "l0_replay.json"
    history = load_json(history_path)
    manifest_sha = sha256(frozen_root / "freeze.json")
    if (history["freeze_sha256"] != manifest_sha or history["gallery"] != 8032
            or history["queries"] != 7614 or history["matched_ranks"] != 38070
            or set(history["methods"]) != set(CORE_METHODS)):
        raise ValueError("Historical L0 witness does not apply to the frozen 8,032-candidate benchmark")
    results = {arm: {} for arm in CONDITIONS}
    for condition in CONDITIONS:
        frozen = FrozenCondition.read(frozen_root, condition)
        expected_q, expected_u, expected_singletons, _ = COUNTS[condition]
        if (len(frozen.gallery) != 8032 or len(frozen.queries) != expected_q
                or len(frozen.texts) != expected_u
                or frozen.manifest["sources"]["inventory"]["sha256"] != ASSETS["inventory"][2]):
            raise ValueError(f"Unexpected frozen denominator for {condition}")
        for method in ROWS:
            if condition != "ct_deepcad" and method == "spectrabrep_ct":
                continue
            result_dir = evaluation / condition / method
            metrics_path = result_dir / "metrics.json"
            if not metrics_path.is_file():
                if condition == "l0" and method in CORE_METHODS:
                    raise ValueError(f"Incomplete historical L0 result for {method}")
                continue
            metrics = load_json(metrics_path)
            metrics_sha = sha256(metrics_path)
            if (metrics["method"] != method or metrics["k"] != list(K_LEVELS)
                    or metrics["gallery"] != 8032 or metrics["frozen_manifest_sha256"] != manifest_sha
                    or metrics["table"] != (2 if condition == "ct_deepcad" else 1)
                    or metrics["domain"] != "DeepCAD" or not metrics["source"]["mode"]):
                raise ValueError(f"Mismatched result metadata for {condition}/{method}")
            if condition == "l0" and method in CORE_METHODS:
                pinned = history["methods"][method]
                if (pinned["metrics_sha256"] != metrics_sha
                        or pinned["query_ranks_sha256"] != metrics["query_ranks_sha256"]):
                    raise ValueError(f"L0 rank receipt differs from the raw replay for {method}")
            else:
                audit_source(metrics, condition)
            rows = read_rank_results(result_dir, frozen)
            singleton = [row for row in rows if frozen.text_counts[row["text_index"]] == 1]
            if len(singleton) != expected_singletons:
                raise ValueError(f"Singleton query axis is incomplete for {condition}/{method}")
            audit_rollup(rows, metrics, 8032, f"{condition}/{method}")
            audit_rollup(singleton, metrics["singleton"], 8032, f"{condition}/{method} singletons")
            known = sum(row["truncated"] is not None for row in rows)
            truncated = sum(row["truncated"] is True for row in rows) if known == len(rows) else None
            historical_external = condition == "l0" and method in CORE_METHODS[:3]
            if (metrics["known_truncation_queries"] != known or metrics["truncated_queries"] != truncated
                    or (not historical_external and known != expected_q)
                    or metrics["failed_queries"] != sum(row["failed"] for row in rows)):
                raise ValueError(f"Failure or truncation counts differ from row evidence for {condition}/{method}")
            results[condition][method] = {
                **metrics,
                "metrics_path": str(metrics_path),
                "metrics_sha256": metrics_sha,
                "query_ranks_path": str(result_dir / "query_ranks.jsonl"),
            }
    comparisons = summarize(frozen_root, evaluation)
    if any(
        set(comparisons["available_core_methods"][arm]) != set(core.keys()) & set(CORE_METHODS)
        for arm, core in results.items()
    ):
        raise ValueError("Evaluation files changed during publication")
    return results, comparisons


def metric_cells(metrics: dict, singleton: bool = False) -> list[str]:
    cells = [f"{percentage:.3f}" for percentage in metrics["recall_percent"]]
    cells += [f"{metrics['median_rank']:g}"]
    if singleton:
        cells += [f"{metrics['singleton']['recall_percent'][-1]:.3f}"]
    return cells


def replace_md_cells(section: str, label: str, replacements: dict[int, str], cell_count: int) -> str:
    pattern = re.compile(r"^\| " + re.escape(label) + r" \|[^\n]+\|$", re.MULTILINE)
    matches = list(pattern.finditer(section))
    if len(matches) != 1:
        raise ValueError(f"Expected one Markdown row for {label}")
    cells = [cell.strip() for cell in matches[0][0].strip("|").split("|")]
    if len(cells) != cell_count:
        raise ValueError(f"Wrong number of Markdown cells for {label}")
    for position, value in replacements.items():
        cells[position] = value
    return pattern.sub(lambda _: "| " + " | ".join(cells) + " |", section)


def existing_or_replace_status(section: str, block: str) -> str:
    section = re.sub(
        rf"\n\n<!-- {STATUS_MARKER}:start -->\n.*?\n<!-- {STATUS_MARKER}:end -->\n\n",
        "\n\n", section, flags=re.DOTALL,
    )
    if not block:
        return section
    return section.rstrip() + (
        f"\n\n<!-- {STATUS_MARKER}:start -->\n{block}\n<!-- {STATUS_MARKER}:end -->\n\n"
    )


def difference_sentence(label: str, contrast: dict) -> str:
    lower, upper = contrast["bootstrap_percentile_95_pp"]
    return (
        f"{label} 的 S₄₈−S₀ 配对 R@20 差值为 **{contrast['difference_percentage_points']:+.3f} 个百分点**，"
        f"新增 {contrast['gained']:,} 个命中，丢失 {contrast['lost']:,} 个命中。"
        f"按 {contrast['groups']:,} 组相同原文抽样 {contrast['repetitions']:,} 次，"
        f"95% 区间为 [{lower:+.3f}, {upper:+.3f}]。"
    )


def render_markdown(source: str, results: dict, comparisons: dict) -> str:
    start = source.index("## 表 1　")
    ct_start = source.index("## 表 2　")
    fusion_start = source.index("## 表 3　")
    prefix, native, ct, suffix = source[:start], source[start:ct_start], source[ct_start:fusion_start], source[fusion_start:]
    for method, (label, _) in ROWS.items():
        native_cells = {2 + level: " / ".join(metric_cells(results[arm][method]))
                        for level, arm in enumerate(CONDITIONS[:4]) if method in results[arm]}
        if native_cells:
            native = replace_md_cells(native, label, native_cells, 6)
        if method in results["ct_deepcad"]:
            ct_cells = dict(enumerate(metric_cells(results["ct_deepcad"][method], singleton=True), start=2))
            ct = replace_md_cells(ct, label, ct_cells, 8)

    native_notes = []
    for arm in CONDITIONS[1:4]:
        if results[arm]:
            methods = "、".join(ROWS[method][0] for method in results[arm])
            native_notes.append(f"{ARM_LABELS[arm]} 已完成 {methods}。")
        if arm in comparisons["s48_minus_sum_r20"]:
            native_notes.append(difference_sentence(ARM_LABELS[arm], comparisons["s48_minus_sum_r20"][arm]))
    if native_notes:
        native_notes[-1] += " [逐方法命中数、截断与失败记录](RETRIEVAL_TABLE_EVIDENCE.json)"

    ct_notes = []
    if results["ct_deepcad"]:
        ct = ct.replace(PENDING_CT, "", 1)
        methods = "、".join(ROWS[method][0] for method in results["ct_deepcad"])
        ct_notes.append(f"表 2 已完成 {methods}。")
        if "ct_deepcad" in comparisons["s48_minus_sum_r20"]:
            ct_notes.append(difference_sentence("CadTranslator–DeepCAD", comparisons["s48_minus_sum_r20"]["ct_deepcad"]))
        pair_panel = comparisons["ct_deepcad_common_uid_panel"]
        paired_methods = [
            ROWS[method][0] for method, item in pair_panel["methods"].items()
            if item["ct_deepcad_minus_native_r20"]
        ]
        if paired_methods:
            methods = "、".join(paired_methods)
            ct_notes.append(f"在 {pair_panel['queries']:,} 个共同目标 UID 上已配对重算 {methods}。")
        ct_notes[-1] += " [唯一原文分子、完整分母与配对结果](RETRIEVAL_TABLE_EVIDENCE.json)"
    return (
        prefix + existing_or_replace_status(native, "\n\n".join(native_notes))
        + existing_or_replace_status(ct, "\n\n".join(ct_notes)) + suffix
    )


def tex_level_cells(cells: list[str]) -> list[list[str]]:
    levels = []
    while cells:
        width = 1 if cells[0] == TEX_PENDING else len(K_LEVELS) + 1
        levels.append(cells[:width])
        cells = cells[width:]
    if len(levels) != 2 or any(len(level) not in (1, 5) or (len(level) == 1 and level[0] != TEX_PENDING)
                               for level in levels):
        raise ValueError("Expected two five-metric level cells per LaTeX row")
    return levels


def render_tex(source: str, results: dict) -> str:
    tables = re.split(r"(?=\\begin\{table\*\}\[tp\])", source)
    if (len(tables) != 4 or r"\label{tab:retrieval-text2cad-levels}" not in tables[1]
            or r"\label{tab:retrieval-cadtranslator-deepcad}" not in tables[2]
            or r"\label{tab:retrieval-cadtranslator-fusion}" not in tables[3]):
        raise ValueError("Unexpected three-table LaTeX structure")
    ending = r" \\"
    for method, (_, label) in ROWS.items():
        pattern = re.compile(r"^" + re.escape(label) + r" & [^\n]+" + re.escape(ending) + r"$", re.MULTILINE)
        matches = list(pattern.finditer(tables[1]))
        if method != "spectrabrep_ct" and len(matches) != 2:
            raise ValueError(f"Expected both native LaTeX rows for {label}")
        for panel, match in reversed(list(enumerate(matches))):
            arms = CONDITIONS[panel * 2:panel * 2 + 2]
            if not any(method in results[arm] for arm in arms):
                continue
            cells = match[0][:-len(ending)].split(" & ")
            levels = tex_level_cells(cells[2:])
            updated = [
                metric_cells(results[arm][method]) if method in results[arm] else levels[i]
                for i, arm in enumerate(arms)
            ]
            line = " & ".join([*cells[:2], *updated[0], *updated[1]]) + ending
            tables[1] = tables[1][:match.start()] + line + tables[1][match.end():]
        if method in results["ct_deepcad"]:
            matches = list(pattern.finditer(tables[2]))
            if len(matches) != 1:
                raise ValueError(f"Expected one CadTranslator LaTeX row for {label}")
            fit = matches[0][0].split(" & ")[1]
            line = " & ".join([label, fit, *metric_cells(results["ct_deepcad"][method], singleton=True)]) + ending
            tables[2] = pattern.sub(lambda _: line, tables[2])
    return "".join(tables)


def update_evidence(source: str, runtime: Path, results: dict, comparisons: dict) -> str:
    evidence = json.loads(source)
    if evidence["base_inputs"]["manifest_sha256"] != ASSETS["inventory"][2]:
        raise ValueError("Source reports are pinned to a different benchmark inventory")
    evidence["benchmark_fill"] = {
        "run_directory": str(runtime),
        "freeze_sha256": sha256(runtime / "frozen" / "freeze.json"),
        "conditions": results,
        "paired_comparisons": comparisons,
        "comparisons_sha256": sha256(runtime / "evaluation" / "comparisons.json"),
        "historical_l0_replay_sha256": sha256(runtime / "evaluation" / "l0_replay.json"),
        "publisher_sha256": sha256(Path(__file__)),
    }
    return json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def decode_report(data: bytes) -> str:
    return data.decode("utf-8-sig").replace("\r\n", "\n")


def encode_report(value: str, original: bytes) -> bytes:
    if b"\r\n" in original:
        value = value.replace("\n", "\r\n")
    return value.encode("utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8")


def write_bundle(output: Path, files: dict[str, bytes]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".retrieval-publication-", dir=output) as staging:
        temporary = Path(staging)
        before = temporary / "before"
        before.mkdir()
        for name, data in files.items():
            target = output / name
            if target.exists():
                (before / name).write_bytes(target.read_bytes())
            (temporary / name).write_bytes(data)
        replaced = []
        try:
            for name in files:
                (temporary / name).replace(output / name)
                replaced.append(name)
        except OSError:
            for name in reversed(replaced):
                if (before / name).exists():
                    (before / name).replace(output / name)
                else:
                    (output / name).unlink()
            raise


def publish(runtime: Path, source_dir: Path, output: Path) -> dict:
    original = {name: (source_dir / name).read_bytes() for name in REPORT_FILES}
    results, comparisons = audit_results(runtime)
    staged_text = {
        MARKDOWN: render_markdown(decode_report(original[MARKDOWN]), results, comparisons),
        TEX: render_tex(decode_report(original[TEX]), results),
        EVIDENCE: update_evidence(decode_report(original[EVIDENCE]), runtime, results, comparisons),
    }
    staged = {name: encode_report(value, original[name]) for name, value in staged_text.items()}
    backup = None
    if source_dir == output:
        if any((output / name).read_bytes() != data for name, data in original.items()):
            raise ValueError("Source tables changed during publication")
        digest = hashlib.sha256(b"".join(original.values())).hexdigest()
        backup = runtime / "publication_backups" / digest
        if not backup.exists():
            write_bundle(backup, original)
        if any((backup / name).read_bytes() != data for name, data in original.items()):
            raise ValueError("Existing report backup is incomplete or inconsistent")
    write_bundle(output, staged)
    return {
        "output": str(output),
        "backup": str(backup) if backup else None,
        "files": {name: sha256(output / name) for name in staged},
        "verified_rows": {arm: list(methods) for arm, methods in results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = publish(args.runtime.resolve(), args.source.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
