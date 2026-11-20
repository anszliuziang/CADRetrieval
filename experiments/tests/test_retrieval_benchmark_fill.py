"""Check the freeze keeps exact text and the scorer receives no target links."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_retrieval_benchmark_fill import (  # noqa: E402
    FrozenCondition,
    external_rows,
    grouped_r20_contrast,
    read_rank_results,
    strict_rank,
    text2cad_rows,
)
from prepare_retrieval_benchmark_fill import (  # noqa: E402
    freeze_condition,
    parse_cadtranslator_texts,
    sha256,
    text_digest,
    write_json,
    write_jsonl,
)
from publish_retrieval_benchmark_fill import (  # noqa: E402
    CONDITIONS,
    PENDING_CT,
    ROWS,
    audit_rollup,
    render_markdown,
    tex_level_cells,
)


class FreezeTests(unittest.TestCase):
    def test_uid_prefix_does_not_parse_the_text(self):
        parsed = parse_cadtranslator_texts(
            ["0000/00000001/ a / b  café   \n", "/0000/00000002/a  b\n"],
            {"0000/00000001", "0000/00000002"},
        )
        self.assertEqual(parsed, {"0000/00000001": "a / b  café", "0000/00000002": "a  b"})

    def test_duplicate_text_links_only_appear_in_queries(self):
        gallery = [
            {"uid": "a", "cad_index": 2, "member": "a.pth"},
            {"uid": "b", "cad_index": 4, "member": "b.pth"},
            {"uid": "c", "cad_index": 7, "member": "c.pth"},
        ]
        texts, queries = freeze_condition(gallery, {"a": "one", "b": "one"})
        self.assertEqual(
            texts, [{"text_index": 0, "text": "one", "text_sha256": text_digest("one")}]
        )
        self.assertEqual(
            queries,
            [
                {"uid": "a", "cad_index": 2, "text_index": 0},
                {"uid": "b", "cad_index": 4, "text_index": 0},
            ],
        )


class EvaluationTests(unittest.TestCase):
    def test_strict_rank_uses_index_order_and_relegates_invalid_candidates(self):
        scores = np.array([9.0, 2.0, 3.0, 2.0, 2.0])
        valid = np.array([False, True, True, True, True])
        self.assertEqual(strict_rank(scores, valid, 3), 3)
        self.assertEqual(strict_rank(scores, valid, 0), 6)

    def test_bootstrap_weights_by_number_of_queries_in_drawn_groups(self):
        before = [
            {"uid": "a", "text_index": 0, "rank": 21},
            {"uid": "b", "text_index": 0, "rank": 21},
            {"uid": "c", "text_index": 1, "rank": 20},
        ]
        after = [dict(row, rank=rank) for row, rank in zip(before, [20, 21, 21], strict=True)]
        actual = grouped_r20_contrast(before, after, repetitions=500, seed=2)
        self.assertEqual((actual["gained"], actual["lost"], actual["both_hits"]), (1, 1, 0))
        self.assertEqual(actual["difference_percentage_points"], 0)
        self.assertEqual(actual["bootstrap_percentile_95_pp"], [-100, 50])

    def toy_condition(self, root):
        gallery = [
            {"uid": str(i), "cad_index": i, "member": f"{i}.pth"} for i in (0, 3, 5, 9)
        ]
        texts, queries = freeze_condition(gallery, {"0": "failed text", "5": "one", "9": "one"})
        names = {"gallery.jsonl": gallery, "texts_l2.jsonl": texts, "queries_l2.jsonl": queries}
        for name, data in names.items():
            write_jsonl(root / name, data)
        np.save(root / "background.npy", np.array([0, 0.6, 0, 0.7], dtype="<f8"))
        manifest = {"domain": "DeepCAD", "files": {name: sha256(root / name)
                                                   for name in [*names, "background.npy"]}}
        write_json(root / "freeze.json", manifest)
        by_text = {i: [row for row in queries if row["text_index"] == i] for i in range(len(texts))}
        counts = Counter({i: len(rows) for i, rows in by_text.items()})
        return FrozenCondition(root, "l2", manifest, gallery, texts, queries, by_text, counts,
                               {row["cad_index"]: i for i, row in enumerate(gallery)})

    def test_dense_and_embedding_inputs_keep_encoding_failures_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self.toy_condition(root)
            gallery = np.array([[1, 0], [0, 1], [0, 1], [0, 0]], dtype="<f8")
            texts = np.array([[0, 1] if row["text"] == "one" else [0, 0] for row in frozen.texts],
                             dtype="<f8")
            method = "duoduoclip"
            for name, axis, values, axis_sha in (
                ("gallery", frozen.gallery, gallery, frozen.gallery_sha),
                ("texts_l2", frozen.texts, texts, frozen.text_sha),
            ):
                is_text = name != "gallery"
                name_path = root / f"{name}.npy"
                np.save(name_path, values)
                status = [
                    {**row, "valid": bool(sum(vector)), "truncated": is_text and bool(sum(vector)),
                     "reason": "toy_encode_error" if not sum(vector) else None}
                    for row, vector in zip(axis, values, strict=True)
                ]
                status_path = root / f"{name}_status.jsonl"
                write_jsonl(status_path, status)
                write_json(root / f"{name}_meta.json", {
                    "weights": {"checkpoint_sha256": "test"},
                    "method": method,
                    "axis_sha256": axis_sha,
                    "vectors_sha256": sha256(name_path),
                    "status_sha256": sha256(status_path),
                })
            dense_path = root / "scores_l2.npy"
            np.save(dense_path, texts @ gallery.T)
            write_json(root / "scores_l2_meta.json", {
                "gallery_sha256": frozen.gallery_sha, "text_sha256": frozen.text_sha,
                "matrix_sha256": sha256(dense_path),
            })
            for encoding in ("embeddings", "scores"):
                result = external_rows(frozen, root, method, encoding, root / encoding)
                self.assertEqual(result["queries"], 3)
                self.assertEqual(result["hits"], [0, 1, 1, 1])
                self.assertEqual(result["median_rank"], 5)
                self.assertEqual(result["failed_queries"], 2)
                self.assertEqual(result["truncated_queries"], 2)
                observed = read_rank_results(root / encoding / frozen.name / method, frozen)
                self.assertEqual([row["rank"] for row in observed], [5, 2, 5])

    def test_text2cad_verifies_bound_shards_and_applies_frozen_background(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = self.toy_condition(root)
            run = root / "run"
            shard_folder = run / "scores/l2"
            shard_folder.mkdir(parents=True)
            freeze_sha = sha256(root / "freeze.json")
            implementation_sha, probe_sha = "1" * 40, "2" * 64
            binding = text_digest("\n".join((implementation_sha, freeze_sha, "l2", probe_sha, "fp32/full272/b16")))
            write_json(run / "run_authorization.json", {
                "mode": "score", "status": "pilot_authorized", "freeze_sha256": freeze_sha,
                "conditions": ["l2"], "approved_sha": implementation_sha, "probe_result_sha256": probe_sha,
                "max_pairs": 8, "max_card_seconds": 500,
            })
            write_json(shard_folder / "text_tokens.json", [
                {"text_index": row["text_index"], "text_sha256": row["text_sha256"],
                 "token_count": 3, "effective_token_count": 3, "truncated": False, "ids_sha256": "3" * 64}
                for row in frozen.texts
            ])
            write_json(run / "l2_complete.json", {
                "condition": "l2", "complete_rows": 2, "gallery": 4, "binding": binding,
                "pairs_charged_total": 8, "cumulative_card_seconds": 1,
                "token_metadata_sha256": sha256(shard_folder / "text_tokens.json"),
            })
            for row in frozen.texts:
                path = shard_folder / f"t{row['text_index']:05d}.npy"
                np.save(path, np.array([-2, -1, -1, -2.5], dtype="<f8"))
                write_json(path.with_suffix(".json"), {
                    "text_index": row["text_index"], "text_sha256": row["text_sha256"],
                    "binding": binding, "scores_sha256": sha256(path),
                })
            metrics = text2cad_rows(frozen, run, root / "results")
            self.assertEqual(metrics["sum"]["hits"], [0, 3, 3, 3])
            self.assertEqual(metrics["s48"]["hits"], [1, 3, 3, 3])
            np.save(shard_folder / "t00000.npy", np.zeros(4, dtype="<f8"))
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                text2cad_rows(frozen, run, root / "results")


class PublishingTests(unittest.TestCase):
    def test_tex_keeps_the_second_pending_level_as_one_cell(self):
        placeholder = r"\multicolumn{5}{c}{---}"
        self.assertEqual(
            tex_level_cells([".4", ".3", ".2", ".1", "1", placeholder]),
            [[".4", ".3", ".2", ".1", "1"], [placeholder]],
        )

    def test_publication_recomputes_counts_with_failed_row_in_denominator(self):
        rows = [{"rank": 1}, {"rank": 4}, {"rank": 6}]
        valid_rollup = {
            "queries": 3,
            "hits": [1, 2, 2, 2],
            "median_rank": 4,
            "recall_percent": [100 / 3, 200 / 3, 200 / 3, 200 / 3],
        }
        audit_rollup(rows, valid_rollup, 5, "toy")
        with self.assertRaisesRegex(ValueError, "does not match"):
            audit_rollup(rows, dict(valid_rollup, hits=[1, 2, 3, 3]), 5, "toy")

    def test_first_ct_cell_removes_the_original_pending_sentence(self):
        label = ROWS["clip"][0]
        source = ("## 表 1　第一张表\n\n## 表 2　第二张表\n\n"
                  f"| {label} | 外部 | — | — | — | — | — | — |\n\n"
                  f"已有参考成绩。{PENDING_CT}\n\n## 表 3　预清单\n")
        results = {condition: {} for condition in CONDITIONS}
        results["ct_deepcad"]["clip"] = {
            "recall_percent": [0.5, 1, 2, 3],
            "median_rank": 10,
            "singleton": {"recall_percent": [0, 1, 2, 3]},
        }
        comparisons = {"s48_minus_sum_r20": {}, "ct_deepcad_common_uid_panel": {"methods": {}}}
        published = render_markdown(source, results, comparisons)
        self.assertNotIn(PENDING_CT, published)
        self.assertIn("| 0.500 | 1.000 | 2.000 | 3.000 | 10 | 3.000 |", published)
        self.assertTrue(published.endswith("## 表 3　预清单\n"))
        self.assertEqual(render_markdown(published, results, comparisons), published)


if __name__ == "__main__":
    unittest.main()
