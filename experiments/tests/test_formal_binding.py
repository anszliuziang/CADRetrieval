import unittest

from experiments.scripts.evaluate_formal_binding import (
    GRID, collision_view, evaluate_view, exact_sign_flip, family_interval, pair_result,
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


if __name__ == "__main__":
    unittest.main()
