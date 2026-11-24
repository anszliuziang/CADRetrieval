import unittest
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.scripts.evaluate_formal_binding import (
    ATTRIBUTES, GRID, collision_view, digest, evaluate_view, exact_sign_flip, family_interval, load_formal_fixture,
    pair_result, summarize_method, verify_review_responses, view_identity_sha,
)


class FormalBindingTests(unittest.TestCase):
    def test_positive_matching_contrast_can_hide_wrong_choices(self):
        outcome = evaluate_view([-2, -3, -8, -8.5])
        self.assertEqual(outcome["D"], 0.5)
        self.assertEqual(outcome["cad_choice"], "same")
        self.assertEqual(outcome["caption_choice"], "same")
        self.assertEqual(outcome["cad_credit_twice"], 2)
        self.assertEqual(outcome["caption_credit_twice"], 2)

    def test_background_cancels_in_columns_and_contrast(self):
        before = evaluate_view([-2, -3, -8, -8.5])
        after = evaluate_view([-2 - 20, -3 - 3, -8 - 20, -8.5 - 3])
        self.assertEqual(before["cad_margins"], after["cad_margins"])
        self.assertEqual(before["D"], after["D"])
        self.assertNotEqual(before["caption_margins"], after["caption_margins"])

    def test_ties_and_missing_cells_are_strict_failures(self):
        tie = evaluate_view([0, 0, 0, 0])
        self.assertEqual(tie["cad_choice"], "tie")
        self.assertEqual(tie["cad_credit_twice"], 2)
        value = evaluate_view([2, None, 0, 2], [None, "text exceeds context", None, None])
        self.assertEqual(value["cad_margins"], [2, None])
        self.assertEqual(value["cad_choice"], "failed")
        self.assertEqual(value["caption_choice"], "failed")
        self.assertEqual(value["cad_credit_twice"], 2)
        self.assertEqual(value["cad_failures"], 1)
        self.assertIsNone(value["D"])
        with self.assertRaisesRegex(ValueError, "failure reason"):
            evaluate_view([2, None, 0, 2])

    def test_all_eight_and_the_same_swap_across_all_eight(self):
        pair = {"pair_id": "p", "kind": "binding",
                "views": [{"order": n, "variant": variant, "first_cut": [["near", "far"][n]] * 2}
                          for n, variant in GRID]}
        correct = evaluate_view([2, 0, 0, 2])
        swapped = evaluate_view([0, 2, 2, 0])
        for selected, expected in ((correct, (1, 0, 0.5)), (swapped, (0, 1, 0.5))):
            grid = dict.fromkeys(GRID, selected)
            row = pair_result(pair, grid)
            self.assertEqual((row["G"], row["G_swap"], row["N"]), expected)
            excluded = pair_result(pair, grid, "indistinguishable_ordered_histories")
            self.assertEqual((excluded["G"], excluded["G_swap"], excluded["N"]), (0, 0, 0))
        grid = dict.fromkeys(GRID, correct)
        grid[0, "near_first"] = swapped
        row = pair_result(pair, grid)
        self.assertEqual((row["G"], row["G_swap"], row["N"]), (0, 0, 0))
        self.assertEqual(row["flip_counts"]["history"]["strict_cad_pass_changes"], 1)
        del grid[0, "near_first"]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            pair_result(pair, grid)

    def test_collision_does_not_invent_unknown_margins(self):
        value = collision_view("caption", {"proof": "identical columns"})
        self.assertEqual(value["scores"], [None] * 4)
        self.assertEqual(value["caption_margins"], [0, 0])
        self.assertEqual(value["cad_margins"], [None, None])
        self.assertEqual(value["cad_credit_twice"], 2)
        self.assertEqual(value["cad_choice"], "tie_or_same")
        self.assertEqual(value["D"], 0)

    def test_family_blocks_use_exact_two_sided_flips(self):
        unanimous = exact_sign_flip([1] * 12)
        self.assertEqual(unanimous, {"extreme": 2, "permutations": 4096, "p_two_sided": 2 / 4096})
        self.assertEqual(exact_sign_flip([1] + [0] * 11)["p_two_sided"], 1)
        self.assertEqual(family_interval([0.25] * 12), [0.25, 0.25])

    def test_pair_scoped_views_join_in_family_summary(self):
        pairs = []
        grid = {}
        views = [{"order": order, "variant": variant, "first_cut": [["near", "far"][order]] * 2}
                 for order, variant in GRID]
        for family in ("f1", "f2"):
            for attribute in ATTRIBUTES:
                pid = f"{family}/{attribute}"
                pairs.append({"family": family, "attribute": attribute, "pair_id": pid, "kind": "binding",
                              "views": views})
                scores = [2, 0, 0, 2] if family == "f1" else [0, 2, 2, 0]
                for order, variant in GRID:
                    grid[pid, order, variant] = evaluate_view(scores)
        details, by_attribute, table = summarize_method("text2cad_release", pairs, grid, {}, "development", False)
        self.assertEqual(details["f1/size_location"]["G"], 1)
        self.assertEqual(details["f2/size_location"]["G_swap"], 1)
        self.assertEqual(by_attribute["size_location"]["families"]["f2"]["order_relation"]["aligned"]["views"], 4)
        self.assertEqual([(row["G"], row["N"], row["difference"]) for row in table], [(0.5, 0.5, 0)] * 2)

    def test_formal_gate_rejects_a_missing_blind_review_ledger(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)

            def record(name, body):
                (root / name).write_bytes(body)
                return {"path": name, "sha256": digest(body)}

            sealed = {
                "scope": "formal", "sealed_at_utc": "2026-11-24T00:00:00Z",
                "families": [f"family-{index}" for index in range(12)],
                "preregistration": record("plan.txt", b"frozen protocol"),
                "pairs": record("pairs.jsonl", b'{"pair_id":"case","kind":"binding"}\n'),
                "histories": record("histories.jsonl", b""),
                "blind_reviews": record("reviews.jsonl", b""),
            }
            path = root / "freeze.json"
            path.write_text(json.dumps(sealed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Blind approvals"):
                load_formal_fixture(path, {})

    def test_stale_approval_is_rejected_even_when_opaque_id_and_choices_are_unchanged(self):
        opaque = "r1234567890abcdef"
        pair = {"pair_id": "test", "review_id": opaque, "kind": "binding"}
        old_item = {"review_id": opaque, "parts": [], "wording_views": [{"text": "previous wording"}]}
        item = {**old_item, "wording_views": [{"text": "new wording for the same part"}]}
        old_packet = digest(json.dumps(old_item).encode())
        packet = digest(json.dumps(item).encode())
        expected = {"V1": {"C1": "P2", "C2": "P1"}}
        approval = {
            "review_id": opaque, "resolution": "accepted", "review_packet_sha256": packet,
            "review_item_sha256": view_identity_sha(item), "reviewers": [
                {"reviewer_id": name, "blind": True, "decision": "approved"} for name in ("reader A", "reader B")
            ],
        }
        responses = {
            name: {"review_id": opaque, "reviewer_id": name, "review_packet_sha256": packet,
                   "review_item_sha256": view_identity_sha(item), "review_date": "2026-11-24",
                   "part_visibility": {part: {"anchor_visible": True, "both_target_holes_visible": True}
                                       for part in ("P1", "P2")},
                   "wording_judgments": [{"view_id": "V1", "caption_match": expected["V1"]}],
                   "equivalent_wording": True}
            for name in ("reader A", "reader B")
        }
        verify_review_responses(pair, item, packet, expected, approval, responses)
        for field, stale in (("review_packet_sha256", old_packet),
                             ("review_item_sha256", view_identity_sha(old_item))):
            with self.subTest(stale=field):
                stale_responses = deepcopy(responses)
                stale_responses["reader B"][field] = stale
                with self.assertRaisesRegex(ValueError, "Blind response is for a different packet or item"):
                    verify_review_responses(pair, item, packet, expected, approval, stale_responses)
                with self.assertRaisesRegex(ValueError, "Coordinator approval is for a different packet or item"):
                    verify_review_responses(pair, item, packet, expected, {**approval, field: stale}, responses)


if __name__ == "__main__":
    unittest.main()
