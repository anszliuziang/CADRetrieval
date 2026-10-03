"""Hand-derived tests for UID ranking and clustered query-weighted statistics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def evaluator():
    path = Path(__file__).parents[1] / 'src/text2cad_g5_cpu.py'
    assert path.is_file(), 'G5 CPU evaluator has not been implemented'
    spec = importlib.util.spec_from_file_location('g5_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_descending_rank_keeps_exact_ties_and_ascending_cad_index():
    # Catches ascending likelihood, unstable ties, epsilon grouping, or 0-based ranks.
    e = evaluator()
    assert e.score_ranks([-3.0, -2.0, -2.0, -2.0 + 1e-12]).tolist() == [4, 2, 3, 1]


@pytest.mark.parametrize('scores', [[], [0, float('nan')], [float('inf')]])
def test_ranking_rejects_empty_or_nonfinite_candidates(scores):
    with pytest.raises(ValueError):
        evaluator().score_ranks(scores)


def test_metrics_use_queries_and_ordinary_even_median():
    # A text-group average would give a different query recall.
    m = evaluator().rank_metrics([1, 1, 4, 8])
    assert m == {'n': 4, 'R@1': .5, 'R@5': .75, 'R@10': 1.0, 'R@20': 1.0,
                 'MedR': 2.5, 'MeanR': 3.5}


def test_weighted_median_repeats_rank_multiplicity():
    e = evaluator()
    assert e.weighted_median([1, 8, 9], [0, 2, 2]) == 8.5
    assert e.weighted_median([1, 8, 9], [2, 0, 0]) == 1.0


def test_cluster_draws_preserve_all_queries_and_share_methods():
    # Draw group0 twice => two rank1 queries; group1 twice => four rank8/9 queries.
    e = evaluator()
    ranks = np.array([[1, 1], [8, 8], [9, 9]])
    r = e.bootstrap_statistics(ranks, [0, 1, 1], [[2, 0], [0, 2]])
    assert r['denominators'].tolist() == [2, 4]
    assert r['R@1'].tolist() == [[1, 1], [0, 0]]
    assert r['MedR'].tolist() == [[1, 1], [8.5, 8.5]]
    assert (r['R@1'][:, 0] - r['R@1'][:, 1]).tolist() == [0, 0]


def test_duplicate_ceilings_use_distinct_targets_and_query_denominator():
    c = evaluator().duplicate_ceilings([1, 7])
    assert c['R@1'] == .25
    assert c['R@5'] == .75
    assert c['R@10'] == 1.0
    assert c['minimum_MedR'] == 3.5  # sorted [1,1,2,3,4,5,6,7]


def toy_linkage():
    import hashlib
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return dict(
        queries=[{'query_id': 'q0', 'text': 'box', 'text_sha256': sha('box'), 'n_chars': 3, 'level': 'L0'},
                 {'query_id': 'q1', 'text': 'box', 'text_sha256': sha('box'), 'n_chars': 3, 'level': 'L0'}],
        gallery=[{'cad_id': 'A', 'cad_seq_member': 'a.pth'}, {'cad_id': 'B', 'cad_seq_member': 'b.pth'}],
        qrels=[{'query_id': 'q0', 'cad_id': 'B', 'relevance': 1}, {'query_id': 'q1', 'cad_id': 'A', 'relevance': 1}],
        texts=[{'text_index': 0, 'text': 'box', 'text_sha256': sha('box'), 'n_chars': 3}],
        store=[{'cad_index': 0, 'member': 'a.pth'}, {'cad_index': 1, 'member': 'b.pth'}],
        lengths=[{'cad_index': 0, 'n_valid_x': 2, 'n_valid_y': 1, 'n_real_tokens': 3},
                 {'cad_index': 1, 'n_valid_x': 1, 'n_valid_y': 1, 'n_real_tokens': 2}],
        mapping={'query_to_text': {'q0': 0, 'q1': 0}, 'cad_index_to_cad_id': ['A', 'B']})


def test_linkage_restores_multiple_original_queries_to_one_text():
    r = evaluator().link_inputs(**toy_linkage())
    assert r['query_groups'].tolist() == [0, 0]
    assert r['positive_cads'].tolist() == [1, 0]
    assert r['lengths'].tolist() == [3, 2]
    assert r['group_sizes'].tolist() == [2]


@pytest.mark.parametrize('breakage', ['missing_qrel', 'duplicate_qrel', 'same_positive', 'wrong_text', 'wrong_member', 'extra_mapping'])
def test_linkage_rejects_incomplete_or_inconsistent_uid_assets(breakage):
    d = toy_linkage()
    if breakage == 'missing_qrel': d['qrels'].pop()
    if breakage == 'duplicate_qrel': d['qrels'].append(dict(d['qrels'][0]))
    if breakage == 'same_positive': d['qrels'][1]['cad_id'] = 'B'
    if breakage == 'wrong_text': d['queries'][1]['text'] = 'sphere'
    if breakage == 'wrong_member': d['store'][1]['member'] = 'a.pth'
    if breakage == 'extra_mapping': d['mapping']['query_to_text']['extra'] = 0
    with pytest.raises(ValueError): evaluator().link_inputs(**d)


def test_length_baselines_are_opposite_and_have_index_ties():
    e = evaluator()
    asc, desc = e.length_ranks([3, 2, 3])
    assert asc.tolist() == [2, 1, 3]
    assert desc.tolist() == [1, 3, 2]


def test_partial_certificate_and_modified_frozen_certificate_rejected(tmp_path):
    import hashlib, json
    e = evaluator()
    p = tmp_path / 'certificate.json'
    p.write_text(json.dumps({'full_matrix_approved': False}))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError): e.require_certificate(p, sha)
    with pytest.raises(ValueError): e.require_certificate(p, '0' * 64)


def test_existing_output_refused_without_overwriting(tmp_path):
    e = evaluator()
    out = tmp_path / 'old_run'; out.mkdir(); (out/'keep').write_text('raw')
    with pytest.raises(FileExistsError): e.new_output(out)
    assert (out/'keep').read_text() == 'raw'


def test_authorization_requires_clean_exact_sha_and_separate_approved_ref(tmp_path):
    import subprocess
    e = evaluator()
    repo = tmp_path / 'code'; repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args]).decode().strip()
    git('init', '-q'); git('config', 'user.name', 'CPU Gate Test')
    git('config', 'user.email', 'cpu-test@example.invalid')
    (repo/'file').write_text('reviewed code')
    git('add', 'file'); git('commit', '-qm', 'reviewed')
    approved = git('rev-parse', 'HEAD')
    (repo/'experiments').mkdir()
    active = repo/'experiments/ACTIVE_TASK.md'
    active.write_text('\n'.join(['- `state`: `pilot_authorized`', f'- `approved_sha`: `{approved}`',
        '- `full_matrix_approved`: true', '- `gpu_budget`: 0', '- `max_scored_pairs`: 0',
        'IDEA01_M1C_G5_CPU_EVALUATION_2026-10-03.md']))
    git('add', 'experiments'); git('commit', '-qm', 'CPU authorization')
    authorization = git('rev-parse', 'HEAD')
    git('checkout', '-q', '--detach', approved)
    e.require_authorization(repo, approved, authorization)
    with pytest.raises(ValueError): e.require_authorization(repo, authorization, authorization)
    with pytest.raises(ValueError): e.require_authorization(repo, approved, approved)
    (repo/'file').write_text('changed')
    with pytest.raises(ValueError): e.require_authorization(repo, approved, authorization)


def test_chunk_reader_checks_hash_dense_identity_and_source_before_ranking(tmp_path):
    import json, hashlib
    e = evaluator()
    header = dict(record_type='chunk_header', shard_id=0, text_range=[0, 1], n_pairs=2,
                  n_scored=2, n_rejected=0, git_sha=e.SCORE_SHA, pair_manifest_sha256='worker',
                  checkpoint_sha256='checkpoint', torch_dtype='float32', trim_mode='none',
                  text_cache=False, score_name='two_head_factorized_logprob_sum',
                  spec_version='text2cad-vector-score-v1')
    rows = [dict(pair_index=0, text_index=0, cad_index=0, status='scored', n_valid_total=3,
                 score_total=-6, mean_total=-2),
            dict(pair_index=1, text_index=0, cad_index=1, status='scored', n_valid_total=2,
                 score_total=-5, mean_total=-2.5)]
    p = tmp_path/'chunk.jsonl'
    def write():
        p.write_text('\n'.join(json.dumps(x) for x in [header,*rows])+'\n')
        return hashlib.sha256(p.read_bytes()).hexdigest()
    h = write()
    total, mean, size = e.load_chunk(p,h,0,2,'worker','checkpoint',[3,2])
    assert total.tolist() == [-6,-5] and mean.tolist() == [-2,-2.5] and size > 0
    with pytest.raises(ValueError): e.load_chunk(p,'0'*64,0,2,'worker','checkpoint',[3,2])
    rows[1]['cad_index'] = 0; h=write()
    with pytest.raises(ValueError): e.load_chunk(p,h,0,2,'worker','checkpoint',[3,2])
    rows[1]['cad_index'] = 1; header['git_sha'] = 'foreign'; h=write()
    with pytest.raises(ValueError): e.load_chunk(p,h,0,2,'worker','checkpoint',[3,2])


def test_empty_layers_have_null_metrics_without_invented_zero_values():
    assert evaluator().rank_metrics([]) == dict(n=0, **{'R@1':None,'R@5':None,'R@10':None,'R@20':None},
                                              MedR=None, MeanR=None)
