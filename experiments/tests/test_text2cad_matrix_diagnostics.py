"""Small hand-derived fixtures for the post-hoc alignment diagnostics."""
import importlib.util
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pytest


def module():
    p = Path(__file__).parents[1] / 'scripts/analyze_text2cad_matrix_20261004.py'
    assert p.is_file(), 'diagnostic code not implemented'
    spec = importlib.util.spec_from_file_location('diag_test', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_rank_direction_exact_ties_and_nonfinite_rejection():
    # Ascending likelihood, epsilon merging, or unstable ties changes these literals.
    m = module()
    assert m.rank_row([-2, -1, -1, -1+1e-12]).tolist() == [4, 2, 3, 1]
    with pytest.raises(ValueError): m.rank_row([0, float('nan')])


def test_group_permutation_reuses_row_and_weights_each_query():
    # Group0 has two targets, group1 one; the second query must not get a new row.
    m = module()
    r = np.array([[1, 2, 3], [3, 1, 2]])
    q = np.array([0, 0, 1]); c = np.array([0, 1, 2])
    assert m.assigned_ranks(r, q, c, [1, 0]).tolist() == [3, 1, 3]
    assert m.metrics([3, 1, 3])['R@1'] == 1/3
    assert m.metrics([3, 1, 3])['MedR'] == 3


def test_exact_wrong_row_expectation_excludes_correct_row():
    m = module()
    r = np.array([[1, 2, 3], [3, 1, 2]])
    out = m.wrong_expectation(r, [0, 0, 1], [0, 1, 2])
    assert out['R@1'] == 1/3
    assert out['MeanR'] == 7/3


def test_variance_decomposition_separates_offsets_and_interaction():
    m = module()
    d = m.decomposition(np.array([[0., 2.], [4., 6.]]))
    assert d['column_ss'] == 4
    assert d['row_ss'] == 16
    assert d['interaction_ss'] == 0
    assert d['column_fraction_after_row_offset'] == 1
    d = m.decomposition(np.array([[1., -1.], [-1., 1.]]))
    assert d['interaction_ss'] == 4
    assert d['column_ss'] == d['row_ss'] == 0


def test_empirical_prior_is_label_free_and_spearman_uses_full_order():
    m = module()
    scores = np.array([[3., 2., 1.], [2., 3., 1.]])
    rr = np.array([[1, 2, 3], [2, 1, 3]])
    d = m.candidate_diagnostics(scores, rr, [2, 1], k=1)
    assert d['prior_ranks'] == [1, 2, 3]
    assert d['topk_overlap_group_mean'] == .5
    assert d['topk_overlap_query_weighted'] == 2/3
    assert d['spearman_group_mean'] == .75
    assert d['topk_unique_candidates'] == 2


def test_score_selectivity_excludes_own_row_and_counts_ties_separately():
    m = module()
    out = m.selectivity(np.array([[2., 0.], [2., 1.], [1., 3.]]), [0, 1], [0, 1])
    assert out['strict_win_fraction'] == [.5, .5]
    assert out['tie_fraction'] == [.5, 0.]


def test_permutation_metrics_share_draws_and_detect_alignment():
    m = module()
    r = np.array([[1, 2], [2, 1]])
    draws = np.array([[0, 1], [1, 0]])
    out = m.permutation_metrics(np.stack([r, r]), [0, 1], [0, 1], draws)
    assert np.array_equal(out[0], out[1])
    assert out[0, :, 0].tolist() == [1., 0.]
    assert out[0, :, 4].tolist() == [1., 2.]


def test_python_source_pin_accepts_only_line_ending_difference(tmp_path):
    # Historical Windows invocation used CRLF, while the Git blob is LF.
    m = module(); p = tmp_path/'helper.py'; b=b'x = 1\n'
    pin=hashlib.sha256(b).hexdigest()
    p.write_bytes(b); lf=m.source_pin(p,pin)
    p.write_bytes(b'x = 1\r\n'); crlf=m.source_pin(p,pin)
    assert lf['canonical_lf_sha256'] == crlf['canonical_lf_sha256'] == pin
    assert lf['actual_sha256'] != crlf['actual_sha256']
    p.write_bytes(b'x = 2\n')
    with pytest.raises(ValueError): m.source_pin(p,pin)


def test_authorization_rejects_self_approval_and_code_change(tmp_path):
    # A caller-provided HEAD hash alone must not authorize a full analysis.
    m=module(); r=tmp_path/'repo'; r.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(r),'-c','user.name=Toy','-c','user.email=toy@example.invalid',*args],text=True).strip()
    git('init','-q'); (r/'experiments/scripts').mkdir(parents=True)
    code=r/'experiments/scripts/analyze_text2cad_matrix_20261004.py'; code.write_text('x=1\n')
    active=r/'experiments/ACTIVE_TASK.md'; active.write_text('- `state`: `review_required`\n')
    git('add','.'); git('commit','-qm','code'); approved=git('rev-parse','HEAD')
    with pytest.raises(ValueError): m.require_authorization(r,approved,approved)
    active.write_text('- `state`: `pilot_authorized`\n- `approved_sha`: `'+approved+'`\n'
                      '- `gpu_budget`: 0 卡小时\n- `max_scored_pairs`: 0\n'
                      '- `active_subplan`: `experiments/plans/IDEA01_TEXT2CAD_EXISTING_MATRIX_DIAGNOSTICS_2026-10-04.md`\n')
    git('add','.'); git('commit','-qm','authorize'); auth=git('rev-parse','HEAD')
    m.require_authorization(r,approved,auth)
    code.write_text('x=2\n'); git('add','.'); git('commit','-qm','wrong code'); wrong=git('rev-parse','HEAD')
    with pytest.raises(ValueError): m.require_authorization(r,approved,wrong)
