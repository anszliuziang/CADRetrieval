"""Independent checks for the frozen E1 scoring and analysis contracts."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def evaluator():
    path = Path(__file__).resolve().parents[1] / "src/idea01/text2cad_e1_eval.py"
    assert path.is_file(), "E1 CPU evaluator has not been implemented"
    from idea01 import text2cad_e1_eval

    return text2cad_e1_eval


def test_background_averages_likelihood_across_48_ordered_rows(evaluator):
    a = np.array([[-820.0, -800.0]] * 24)
    b = np.array([[-820.0, -810.0]] * 24)
    result = evaluator.score_readouts(np.array([[-810.0, -806.0]]), np.vstack((a, b)))
    assert result["S48"][0].tolist() == pytest.approx([10, -5.30689821834])
    assert result["half_0_23"][0].tolist() == pytest.approx([10, -6])
    assert result["half_24_47"][0].tolist() == pytest.approx([10, 4])


def test_dimension_direction_preposition_does_not_mean_inches(evaluator):
    rules = evaluator.numeric_rules()
    parsed = evaluator.query_numeric_values(
        "extrude to a depth of 0.75 in the direction of the normal", rules,
    )
    assert parsed["physical"] == set()
    assert parsed["normalized"] == {("depth", "normalized", 750)}
    physical = evaluator.query_numeric_values("use a depth of 0.75 in.", rules)
    assert physical["physical"]


def test_literal_baseline_excludes_unit_converted_candidate_values(evaluator):
    result = evaluator.candidate_numeric_values([
        {"attribute": "depth", "value_space": "normalized", "value": 0.75},
        {"attribute": "depth", "value_space": "physical", "value": 19.05, "unit": "mm"},
    ], evaluator.numeric_rules())
    assert result["physical"]
    assert result["literal"] == {0.75}


def test_ranks_tie_by_hash_and_preserve_small_nonzero_differences(evaluator):
    uids = ["a", "b", "c"]
    tie_order = sorted(
        (0, 1), key=lambda i: hashlib.sha256(f"cad-domain-v1|tie-cad|fusion360|{uids[i]}".encode()).digest()
    )
    expected = [0, 0, 1]
    expected[tie_order[0]], expected[tie_order[1]] = 2, 3
    result = evaluator.rank_matrix(np.array([[-3.0, -3.0, -3.0 + 1e-12]]), uids, "fusion360")
    assert result.tolist() == [expected]


def test_b_gate_uses_paired_discordances_and_97_5_percent_marginals(evaluator):
    identical = evaluator.encoding_loss(np.full(256, 20), np.full(256, 20))
    assert identical["decision"] == "excludes_net_loss_above_tolerance"
    assert identical["ci95_pp"] == pytest.approx([-1.697161, 1.697161], abs=1e-5)
    reencoded = np.array([21] * 32 + [20] * 224)
    official = np.array([20] * 255 + [21])
    changed = evaluator.encoding_loss(reencoded, official)
    assert (changed["official_only"], changed["reencoded_only"]) == (32, 1)
    assert changed["net_loss_pp"] == pytest.approx(12.109375)
    assert changed["decision"] == "encoding_degradation"


def test_permutation_uses_common_draws_and_holm(evaluator):
    ranks = np.array([[1, 2], [2, 1]])
    result = evaluator.permutation_test(
        {"S0": ranks, "S48": ranks}, np.array([0, 1]), "fusion360", draws=90, k=1
    )
    native, adjusted = result["methods"]["S0"], result["methods"]["S48"]
    assert native == adjusted
    assert native["hits"] == 2
    assert native["p"] == pytest.approx((1 + native["permutations_ge_observed"]) / 91)
    assert native["holm_p"] == pytest.approx(min(1, 2 * native["p"]))
    assert result["draws"] == 90


def test_paired_bootstrap_shares_group_draws(evaluator):
    result = evaluator.paired_ranks([1, 21, 20], [21, 1, 20], ["same", "same", "other"], "deepcad", "toy")
    assert (result["added_hits"], result["lost_hits"], result["net_pp"]) == (1, 1, 0)
    assert result["ci95_pp"] == [0, 0]


def test_replacing_columns_also_replaces_every_background_contribution(evaluator):
    raw = np.array([[-4.0, -7.0, -9.0], [-2.0, -5.0, -10.0]])
    bg = np.broadcast_to([-6.0, -2.0, -8.0], (48, 3)).copy()
    patched_raw, patched_bg = evaluator.patch_columns(
        raw, bg, ["a", "c", "b"], np.array([[-1.0], [-3.0]]), np.full((48, 1), -10.0), ["c"]
    )
    assert patched_raw.tolist() == [[-4, -1, -9], [-2, -3, -10]]
    assert patched_bg[-1].tolist() == [-6, -10, -8]
    assert evaluator.score_readouts(patched_raw, patched_bg)["S48"].tolist() == [[2, 9, -1], [4, 7, -2]]
    assert (raw[:, 1] == [-7, -5]).all() and (bg[:, 1] == -2).all()


def test_difficult_pairs_are_disjoint_and_double_difference_is_offset_invariant(evaluator):
    matrix = np.array([[-8.0, -9.0, -20.0], [-10.0, -7.0, -20.0], [-15.0, -21.0, -3.0]])
    gallery = [
        dict(uid=str(i), group_id=str(i), n_valid=n, program_features={
            "has_single_circle": True, "has_near_rectangle": False, "has_multiloop_or_cut": False,
        })
        for i, n in enumerate([2, 2, 3])
    ]
    raw_ranks = evaluator.rank_matrix(matrix, ["0", "1", "2"], "deepcad")
    corrected = matrix - np.array([-10.0, -2.0, -3.0])
    adjusted_ranks = evaluator.rank_matrix(corrected, ["0", "1", "2"], "deepcad")
    result = evaluator.difficult_pairs(gallery, matrix, corrected, raw_ranks, adjusted_ranks, "deepcad", [])
    assert result["exact"]["n_pairs"] == 1
    pair = result["exact"]["pairs"][0]
    assert (pair["D"], pair["D_corrected"], pair["pair_hit"]) == (4, 4, 1)
    assert pair["both_correct"] == {"S0": True, "S48": False}
    assert result["relaxed"]["n_pairs"] == 0


def test_numeric_parser_keeps_units_and_value_spaces_separate(evaluator):
    rules = evaluator.numeric_rules()
    query = evaluator.query_numeric_values("Two holes with a diameter of 1 cm, a depth of 0.125, and width 2:1.", rules)
    program = evaluator.candidate_numeric_values([
        {"attribute": "hole", "value_space": "count", "value": 2},
        {"attribute": "diameter", "value_space": "physical", "value": 10, "unit": "mm"},
        {"attribute": "depth", "value_space": "normalized", "value": 0.1251},
        {"attribute": "width", "value_space": "ratio", "value": 2},
    ], rules)
    assert query["count"] == program["count"] == {("hole", "integer", 2)}
    assert query["physical"] == program["physical"] == {("diameter", "mm", 100)}
    assert query["normalized"] == program["normalized"] == {("depth", "normalized", 125)}
    assert query["ratio"] == program["ratio"] == {("width", "ratio", 200)}


def test_source_only_gate_opens_no_c_files(tmp_path):
    source, scores, output = (tmp_path / name for name in ("input", "scores", "out"))
    source.mkdir()
    scores.mkdir()
    n = 256
    uids = [f"part{i:03}" for i in range(n)]
    gallery = [dict(uid=uid, group_id=f"g{i:03}", text=f"part {i}", token_count=3, n_valid=275)
               for i, uid in enumerate(uids)]
    (source / "freeze.json").write_text("{}", encoding="utf-8")
    (source / "gallery_deepcad.jsonl").write_text("".join(json.dumps(row) + "\n" for row in gallery), encoding="utf-8")
    np.savez(scores / "b.npz", query_uids=uids, cad_uids=uids, scores=-np.ones((n, n)), background=-np.ones((48, n)))
    np.savez(scores / "b_official.npz", query_uids=uids, cad_uids=uids, scores=-np.ones((n, n)), background=-np.ones((48, n)))
    (scores / "c.npz").write_text("deliberately corrupt", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/eval_text2cad_e1.py"
    assert script.is_file(), "E1 CLI has not been implemented"
    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--scores", str(scores), "--output", str(output), "--source-only"],
        capture_output=True, encoding="utf-8", check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert result["encoding_check"]["status"] == "pass"
    assert result["encoding_check"]["methods"]["S0"]["finite_gallery_net_loss_pp"] == 0
    assert result["encoding_check"]["methods"]["S48"]["n_rank_changed"] == 0
    provenance = json.loads((output / "score_manifest.json").read_text(encoding="utf-8"))
    assert set(provenance["archives"]) == {"b", "b_official"}
    assert (output / "RESULTS.md").read_text(encoding="utf-8").startswith("# E1 retrieval evaluation")
    sums = (output / "SHA256SUMS").read_text(encoding="utf-8")
    expected = hashlib.sha256((output / "ranks.parquet").read_bytes()).hexdigest()
    assert f"{expected}  ranks.parquet" in sums


def test_full_run_keeps_c_sealed_when_only_corrected_encoding_gate_fails(tmp_path, evaluator):
    source, scores, output = (tmp_path / name for name in ("input", "scores", "out"))
    source.mkdir()
    scores.mkdir()
    n = 256
    uids = [str(i) for i in range(n)]
    gallery = [dict(uid=uid, group_id=uid, text=f"part {uid}") for uid in uids]
    (source / "freeze.json").write_text("{}", encoding="utf-8")
    (source / "gallery_deepcad.jsonl").write_text("".join(json.dumps(row) + "\n" for row in gallery), encoding="utf-8")
    raw = np.full((n, n), -10.0)
    np.fill_diagonal(raw, -8)
    raw[:, :20] = -5
    background = np.full((48, n), -5.0)
    official_background = background.copy()
    official_background[:, :20] = -1
    for arm, bg in (("b", background), ("b_official", official_background)):
        np.savez(scores / f"{arm}.npz", query_uids=uids, cad_uids=uids, scores=raw, background=bg)
    (scores / "c.npz").write_text("sealed and deliberately corrupt", encoding="utf-8")

    metrics = evaluator.evaluate(source, scores, output)
    assert metrics["phase"] == "source_encoding_degradation"
    assert metrics["encoding_check"]["methods"]["S0"]["status"] == "pass"
    assert metrics["encoding_check"]["methods"]["S48"]["status"] == "fail"
    assert metrics["encoding_check"]["methods"]["S48"]["official_only"] == 236


def test_load_scores_rejects_reordered_columns(tmp_path, evaluator):
    scores = tmp_path / "b.npz"
    np.savez(scores, query_uids=["p"], cad_uids=["p", "q"], scores=np.ones((1, 2)), background=np.ones((48, 2)))
    with pytest.raises(ValueError, match="CAD axis"):
        evaluator.load_scores(scores, [{"uid": "p"}], ["q", "p"])


def test_blind_brep_preserves_geometry_only_occt_export_and_masks_filename(tmp_path, evaluator):
    source = tmp_path / "source"
    public = tmp_path / "public"
    for d in (source, public):
        d.mkdir()
    geometry = b"DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\nLocations 0\n"
    uid = "100155_57ec5fc6_0000"
    (source / f"{uid}.brep").write_bytes(geometry)
    path = evaluator.blind_asset(source, {"uid": uid, "geometry_relpath": f"{uid}.brep"}, public, "opaque")
    assert path == "assets/opaque.brep"
    assert (public / path).read_bytes() == geometry
    (source / f"{uid}.brep").write_bytes(geometry + f"Name {uid}\n".encode())
    assert evaluator.blind_asset(source, {"uid": uid, "geometry_relpath": f"{uid}.brep"}, public, "unsafe") is None


def test_near_training_sensitivity_reranks_retained_queries_against_smaller_gallery(evaluator):
    n = 22
    gallery = [dict(uid=f"g{i}", group_id=f"g{i}", text="a part", source_train_near_status="no_corroboration") for i in range(n)]
    gallery[1]["source_train_near_status"] = "suspected"
    gallery[2]["source_train_near_status"] = "suspected"
    gallery[3].pop("source_train_near_status")
    raw = np.full((n, n), -10.0)
    np.fill_diagonal(raw, -2)
    raw[0, 0] = -12
    before = np.diag(evaluator.rank_matrix(raw, [r["uid"] for r in gallery], "fusion360"))
    assert before[0] == 22
    result = evaluator.exclude_suspected(gallery, gallery, raw, raw + 10, "fusion360")
    assert result["n_suspected_excluded"] == 2
    assert result["n_status_missing_or_unscreened_retained"] == 1
    assert result["n_retained_queries"] == 20 and result["n_retained_gallery"] == 20
    assert result["methods"]["S0"]["hits"]["R@20"] == 20


def test_full_run_reranks_untouched_targets_after_column_replacement(tmp_path, evaluator):
    source, scores, output = (tmp_path / name for name in ("input", "scores", "out"))
    source.mkdir()
    scores.mkdir()
    n = 256
    a_uids = [f"d{i:03}" for i in range(n)]
    c_uids = [f"f{i:03}" for i in range(n)]
    gallery = lambda uids: [dict(uid=uid, group_id=f"g{i:03}", text=f"part {i}", token_count=3, n_valid=275,
                                 n_extrusions=1, l0_text="shape" if i == 0 else "", token_ids=[101, i, 102]) for i, uid in enumerate(uids)]

    def records(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    b_rows, c_rows = gallery(a_uids), gallery(c_uids)
    records(source / "gallery_deepcad.jsonl", b_rows)
    records(source / "gallery_fusion360.jsonl", c_rows)
    records(source / "gallery_fusion_original.jsonl", c_rows[:1])
    records(source / "queries_a.jsonl", [{**b_rows[0], "text": "shape"}])
    records(source / "queries_c.jsonl", c_rows)
    diag = sorted(c_rows, key=lambda r: hashlib.sha256(
        f"cad-domain-v1|diagnostic|fusion360|{r['group_id']}".encode()).digest())[:48]
    chosen = diag[0]
    rewrite = [dict(**{**chosen, "text": "new description"}, variant="Qgeom", status="changed_valid", n_constraints=2)]
    records(source / "queries_c_rewrite.jsonl", rewrite)
    (source / "freeze.json").write_text(json.dumps({"replacement_uids": c_uids[:1], "diagnostic_uids": [r["uid"] for r in diag]}), encoding="utf-8")
    (source / "baseline_reference.json").write_text(json.dumps({"token_counts": [1, 4, 8], "n_valid": [100, 275, 400], "n_extrusions": [1, 2, 3]}), encoding="utf-8")
    raw = np.full((n, n), -10.0)
    np.fill_diagonal(raw, -2)
    background = np.full((48, n), -10.0)
    for arm in ("b", "b_official", "c", "a"):
        uids = c_uids if arm == "c" else a_uids
        rows = raw.copy()
        if arm == "c":
            rows[0, 0] = -12
        if arm == "a":
            uids, rows = uids[:1], rows[:1]
        np.savez(scores / f"{arm}.npz", query_uids=uids, cad_uids=c_uids if arm == "c" else a_uids,
                 scores=rows, background=background)
    alternative = np.full((n, 1), -30.0)
    alternative[:2] = -1
    np.savez(scores / "c_alt.npz", query_uids=c_uids, cad_uids=c_uids[:1], scores=alternative,
             background=np.full((48, 1), -20.0))
    rewrite_scores = np.full((1, n), -10.0)
    rewrite_scores[0, c_uids.index(chosen["uid"])] = -20
    np.savez(scores / "c_rewrite.npz", query_uids=[chosen["uid"]], query_variants=["Qgeom"], cad_uids=c_uids,
             scores=rewrite_scores)

    result = evaluator.evaluate(source, scores, output)
    assert result["primary_C"]["methods"]["S0"]["hits"] == 255
    assert result["c_encoding_sensitivity"]["methods"]["S0"]["target_replaced"]["added_hits"] == 1
    assert result["c_encoding_sensitivity"]["methods"]["S0"]["target_unreplaced"]["n_rank_changed"] == 1
    assert result["c_encoding_sensitivity"]["methods"]["S48"]["target_unreplaced"]["n_rank_changed"] == 1
    assert result["main"]["B"]["surface_baselines"]["length"]["n_all_candidates_equal"] == 256
    assert result["c_rewrites"]["variants"]["Qgeom"]["statuses"]["changed_valid"] == 1
    assert result["c_rewrites"]["variants"]["Qgeom"]["methods"]["S0"]["lost_hits"] == 1
    assert result["semantic_review"]["status"] == "pending_two_independent_annotators"
    packets = [json.loads(line) for line in (output / "blind_packets/public/packets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(packets) == 48
    assert all(len(p["candidates"]) <= 8 for p in packets)
    assert all("uid" not in item and "rank" not in item for p in packets for item in p["candidates"])
