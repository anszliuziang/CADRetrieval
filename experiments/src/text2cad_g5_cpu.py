"""Frozen G5 exact-UID evaluation. Only stdlib and NumPy; no model imports."""
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

KS = (1, 5, 10, 20)
METHODS = ('sum', 'mean', 'length_asc', 'length_desc', 'random')
SCORE_SHA = '56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719'
MANIFEST_SHA = '97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733'
CERTIFICATE_SHA = 'afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6'
SEED = 20261003
REPLICATES = 5000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8', newline='\n')


def score_ranks(scores):
    values = np.asarray(scores, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all(), 'invalid candidate scores')
    order = np.lexsort((np.arange(values.size), -values))
    ranks = np.empty(values.size, dtype=np.int32)
    ranks[order] = np.arange(1, values.size + 1)
    return ranks


def length_ranks(lengths):
    a = np.asarray(lengths)
    require(a.ndim == 1 and a.size > 0 and np.isfinite(a).all() and (a > 0).all(), 'invalid lengths')
    return score_ranks(-a), score_ranks(a)


def valid_ranks(ranks):
    a = np.asarray(ranks)
    require(a.ndim == 1 and np.isfinite(a).all() and (a > 0).all()
            and (a == np.floor(a)).all(), 'invalid positive ranks')
    return a.astype(np.int64)


def rank_metrics(ranks):
    a = valid_ranks(ranks)
    if not len(a):
        return dict(n=0, **{f'R@{k}': None for k in KS}, MedR=None, MeanR=None)
    return dict(n=len(a), **{f'R@{k}': float(np.mean(a <= k)) for k in KS},
                MedR=float(np.median(a)), MeanR=float(np.mean(a)))


def median_sorted(ranks, weights):
    cumulative = np.cumsum(weights, dtype=np.int64)
    n = int(cumulative[-1])
    require(n > 0, 'empty weighted sample')
    lo = np.searchsorted(cumulative, (n + 1) // 2, side='left')
    hi = np.searchsorted(cumulative, n // 2 + 1, side='left')
    return float((ranks[lo] + ranks[hi]) / 2.0)


def weighted_median(ranks, weights):
    a = valid_ranks(ranks)
    w = np.asarray(weights)
    require(w.shape == a.shape and np.isfinite(w).all() and (w >= 0).all()
            and (w == np.floor(w)).all() and w.sum() > 0, 'invalid sample weights')
    order = np.argsort(a, kind='stable')
    return median_sorted(a[order], w[order])


def bootstrap_statistics(rank_matrix, query_groups, draw_counts):
    ranks = np.asarray(rank_matrix)
    groups = np.asarray(query_groups)
    counts = np.asarray(draw_counts)
    require(ranks.ndim == 2 and len(ranks) > 0 and ranks.shape[1] > 0
            and np.isfinite(ranks).all() and (ranks > 0).all()
            and (ranks == np.floor(ranks)).all(), 'bootstrap ranks invalid')
    require(groups.shape == (len(ranks),) and np.isfinite(groups).all()
            and (groups == np.floor(groups)).all() and (groups >= 0).all(), 'bootstrap groups invalid')
    groups = groups.astype(np.int64)
    g = int(groups.max()) + 1
    require(np.array_equal(np.unique(groups), np.arange(g)), 'bootstrap groups must be dense/nonempty')
    require(counts.ndim == 2 and len(counts) > 0 and counts.shape[1] == g
            and np.isfinite(counts).all() and (counts >= 0).all()
            and (counts == np.floor(counts)).all() and (counts.sum(axis=1) == g).all(), 'cluster draws invalid')
    counts = counts.astype(np.int64)
    group_sizes = np.bincount(groups, minlength=g)
    denominators = counts @ group_sizes
    result = {'denominators': denominators}
    for k in KS:
        hits = np.zeros((g, ranks.shape[1]), dtype=np.int64)
        np.add.at(hits, groups, ranks <= k)
        result[f'R@{k}'] = (counts @ hits) / denominators[:, None]
    medians = np.empty((len(counts), ranks.shape[1]))
    for method in range(ranks.shape[1]):
        order = np.argsort(ranks[:, method], kind='stable')
        sorted_ranks = ranks[order, method]
        sorted_groups = groups[order]
        for j, draw in enumerate(counts):
            medians[j, method] = median_sorted(sorted_ranks, draw[sorted_groups])
    result['MedR'] = medians
    return result


def duplicate_ceilings(group_sizes):
    sizes = np.asarray(group_sizes)
    require(sizes.ndim == 1 and len(sizes) > 0 and (sizes > 0).all()
            and (sizes == np.floor(sizes)).all(), 'invalid positive group sizes')
    n = int(sizes.sum())
    lower = np.concatenate([np.arange(1, int(s) + 1) for s in sizes])
    return {**{f'R@{k}': float(np.minimum(sizes, k).sum() / n) for k in KS},
            'minimum_MedR': float(np.median(lower)), 'n_queries': n, 'n_groups': len(sizes)}


def unique_by(rows, key):
    d = {}
    for row in rows:
        require(row[key] not in d, 'duplicate ' + key)
        d[row[key]] = row
    return d


def link_inputs(queries, gallery, qrels, texts, store, lengths, mapping):
    q = unique_by(queries, 'query_id')
    c = unique_by(gallery, 'cad_id')
    tx = unique_by(texts, 'text_index')
    st = unique_by(store, 'cad_index')
    ln = unique_by(lengths, 'cad_index')
    require(len(q) > 0 and len(c) > 0 and len(tx) > 0, 'empty UID inputs')
    require(set(tx) == set(range(len(tx))) and set(st) == set(ln) == set(range(len(c))), 'dense input indices')
    cad_ids = mapping['cad_index_to_cad_id']
    require(cad_ids == sorted(c) and len(set(cad_ids)) == len(c), 'gallery UID order differs')
    require(set(mapping['query_to_text']) == set(q), 'query mapping coverage differs')
    for i, cad_id in enumerate(cad_ids):
        require(c[cad_id]['cad_seq_member'] == st[i]['member'], 'CAD member linkage differs')
    for i in range(len(tx)):
        t = tx[i]['text']
        require(t and t == t.strip() and tx[i]['text_sha256'] == hashlib.sha256(t.encode()).hexdigest()
                and tx[i]['n_chars'] == len(t), 'worker text identity')
    require([(tx[i]['text_sha256'], tx[i]['text']) for i in range(len(tx))]
            == sorted({(r['text_sha256'], r['text']) for r in texts}), 'unique canonical worker text order')
    rel = unique_by(qrels, 'query_id')
    require(set(rel) == set(q), 'qrels coverage differs')
    cad_index = {cid: i for i, cid in enumerate(cad_ids)}
    query_groups = []
    positives = []
    group_targets = [set() for _ in tx]
    for row in queries:
        qid = row['query_id']; t = row['text'].strip(); ti = mapping['query_to_text'][qid]
        require(type(ti) is int and ti in tx and row['level'] == 'L0' and t
                and t == tx[ti]['text'] and row['text_sha256'] == hashlib.sha256(t.encode()).hexdigest()
                and row['n_chars'] == len(row['text']), 'query text/level linkage differs')
        target = rel[qid]
        require(target['relevance'] == 1 and target['cad_id'] in c, 'invalid UID positive')
        ci = cad_index[target['cad_id']]
        require(ci not in group_targets[ti], 'same UID repeated within identical-text group')
        group_targets[ti].add(ci); query_groups.append(ti); positives.append(ci)
    sizes = np.bincount(query_groups, minlength=len(tx))
    require((sizes > 0).all(), 'unqueried text group')
    valid_lengths = []
    for i in range(len(c)):
        row = ln[i]
        require(all(type(row[k]) is int and row[k] >= 0 for k in ('n_valid_x', 'n_valid_y', 'n_real_tokens'))
                and row['n_valid_x'] + row['n_valid_y'] > 0, 'invalid pinned lengths')
        valid_lengths.append(row['n_valid_x'] + row['n_valid_y'])
    return dict(query_groups=np.asarray(query_groups, dtype=np.int32),
                positive_cads=np.asarray(positives, dtype=np.int32), group_sizes=sizes,
                lengths=np.asarray(valid_lengths, dtype=np.int32), cad_ids=cad_ids,
                query_ids=[r['query_id'] for r in queries])


def require_certificate(path, expected_sha):
    require(sha256(path) == expected_sha, 'final matrix certificate byte pin differs')
    c = read_json(path)
    require(c.get('schema') == 'text2cad-g4-final-matrix-certificate-v1'
            and c.get('full_matrix_approved') is True and c.get('score_source_sha') == SCORE_SHA
            and c.get('manifest_sha256') == MANIFEST_SHA and c.get('text_count') == 3644
            and c.get('cad_count') == 8046 and c.get('full_score_rows_audited') == 29319624
            and c.get('payload_files') == 3666 and c.get('payload_bytes') == 14858752230,
            'unapproved/partial or foreign matrix certificate')
    return c


def new_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def git(repo, *args):
    r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, check=True)
    return r.stdout.decode('utf-8').strip()


def require_authorization(repo, approved_sha, authorization_ref):
    require(re.fullmatch('[0-9a-f]{40}', approved_sha) is not None
            and re.fullmatch('[0-9a-f]{40}', authorization_ref) is not None, 'full commit SHA required')
    require(git(repo, 'rev-parse', 'HEAD') == approved_sha, 'checkout must equal approved G5 code SHA')
    require(git(repo, 'status', '--porcelain', '--untracked-files=all') == '', 'CPU code checkout must be clean')
    try:
        git(repo, 'merge-base', '--is-ancestor', approved_sha, authorization_ref)
        active = git(repo, 'show', authorization_ref + ':experiments/ACTIVE_TASK.md')
    except subprocess.CalledProcessError as exc:
        raise ValueError('authorization ref missing or not descended from approved code') from exc
    for text in ('- `state`: `pilot_authorized`', '- `approved_sha`: `' + approved_sha + '`',
                 '- `full_matrix_approved`: true', '- `gpu_budget`: 0',
                 'IDEA01_M1C_G5_CPU_EVALUATION_2026-10-03.md'):
        require(text in active, 'CPU authorization missing ' + text)
    require('- `max_scored_pairs`: 0' in active, 'new scoring not authorized')


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def manifest_entries(path):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, 'G4 raw manifest differs')
    require(b'\r' not in raw and raw.endswith(b'\n'), 'G4 manifest must be LF')
    entries = {}
    for line in raw.decode().splitlines():
        digest, name = line.split('  ', 1)
        name = Path(name).as_posix()
        require(name not in entries and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'duplicate/escaping manifest path')
        entries[name] = digest
    require(len(entries) == 3666, 'G4 manifest incomplete')
    return entries


def load_chunk(path, expected_hash, text_index, n_cad, worker_pair_hash, checkpoint_hash, lengths):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_hash, 'raw chunk changed ' + str(path))
    lines = raw.splitlines()
    require(len(lines) == n_cad + 1, 'chunk row coverage')
    header = json.loads(lines[0]); shard = text_index // 1822
    wanted = dict(record_type='chunk_header', shard_id=shard, text_range=[text_index, text_index+1],
                  n_pairs=n_cad, n_scored=n_cad, n_rejected=0, git_sha=SCORE_SHA,
                  pair_manifest_sha256=worker_pair_hash, checkpoint_sha256=checkpoint_hash,
                  torch_dtype='float32', trim_mode='none', text_cache=False,
                  score_name='two_head_factorized_logprob_sum', spec_version='text2cad-vector-score-v1')
    require(all(header.get(k) == v for k, v in wanted.items()), 'chunk source/spec header differs')
    total = np.empty(n_cad, dtype=np.float64); mean = np.empty(n_cad, dtype=np.float64)
    for ci, line in enumerate(lines[1:]):
        row = json.loads(line)
        require(row['pair_index'] == text_index*n_cad + ci and row['text_index'] == text_index
                and row['cad_index'] == ci and row['status'] == 'scored'
                and row['n_valid_total'] == int(lengths[ci]), 'raw identity/length differs')
        total[ci] = row['score_total']; mean[ci] = row['mean_total']
    require(np.isfinite(total).all() and np.isfinite(mean).all(), 'nonfinite raw score')
    return total, mean, len(raw)


def clustered_intervals(rank_matrix, groups):
    g = int(np.max(groups)) + 1
    rng = np.random.default_rng(SEED)
    counts = np.empty((REPLICATES, g), dtype=np.int16)
    for i in range(REPLICATES):
        counts[i] = np.bincount(rng.integers(0, g, size=g), minlength=g)
    samples = bootstrap_statistics(rank_matrix, groups, counts)
    ci = {method: {} for method in METHODS}
    paired = {method: {} for method in ('length_asc', 'length_desc', 'random')}
    for metric in [*(f'R@{k}' for k in KS), 'MedR']:
        for j, method in enumerate(METHODS):
            ci[method][metric] = np.quantile(samples[metric][:, j], [.025, .975]).tolist()
    for method in paired:
        j = METHODS.index(method)
        for k in KS:
            paired[method][f'R@{k}'] = np.quantile(samples[f'R@{k}'][:, 0] - samples[f'R@{k}'][:, j], [.025, .975]).tolist()
    return dict(method_ci95=ci, paired_sum_gain_ci95=paired, replicates=REPLICATES,
                seed=SEED, resampling_unit='unique_text_group', query_weighted=True,
                shared_draws=True, percentile=[.025, .975], draw_counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
                denominator_min=int(samples['denominators'].min()), denominator_max=int(samples['denominators'].max()))


def stratified_metrics(rank_matrix, groups, sizes, positive_cads, lengths):
    qs = sizes[groups]
    masks = {'size_1': qs == 1, 'size_2_5': (qs >= 2) & (qs <= 5),
             'size_6_20': (qs >= 6) & (qs <= 20), 'size_21_100': (qs >= 21) & (qs <= 100),
             'size_gt100': qs > 100, 'unique': qs == 1, 'repeated': qs > 1}
    cuts = np.quantile(lengths, [.25, .5, .75])
    bins = np.searchsorted(cuts, lengths[positive_cads], side='left')
    masks.update({f'length_quartile_{i+1}': bins == i for i in range(4)})
    return dict(gallery_length_quartile_cutpoints=cuts.tolist(), length_boundary_rule='lower bin includes exact boundary',
                strata={name: {method: rank_metrics(rank_matrix[mask, j]) for j, method in enumerate(METHODS)}
                        for name, mask in masks.items()})


def run_evaluation(repo, data_root, runtime_root, output, approved_sha, authorization_ref):
    repo, data_root, runtime_root, output = map(Path, (repo, data_root, runtime_root, output))
    require_authorization(repo, approved_sha, authorization_ref)
    config_path = repo / 'experiments/configs/text2cad_g5_cpu_frozen_20261003.json'
    config = read_json(config_path)
    require(config['certificate_sha256'] == CERTIFICATE_SHA and config['seed'] == SEED
            and config['bootstrap_replicates'] == REPLICATES and config['methods'] == list(METHODS), 'frozen protocol differs')
    certificate_path = repo / config['certificate_path']
    certificate = require_certificate(certificate_path, CERTIFICATE_SHA)
    require(not output.exists(), 'existing output directory')
    inputs = {name: (data_root if pin['root'] == 'data' else runtime_root) / pin['path']
              for name, pin in config['inputs'].items()}
    fingerprints = {}
    for name, path in inputs.items():
        digest = sha256(path)
        require(digest == config['inputs'][name]['sha256'], 'frozen input SHA differs ' + name)
        fingerprints[name] = dict(path=str(path), sha256=digest, bytes=path.stat().st_size)
    linked = link_inputs(**{name: jsonl(inputs[name]) for name in ('queries', 'gallery', 'qrels', 'texts', 'store', 'lengths')},
                         mapping=read_json(inputs['mapping']))
    require(len(linked['query_ids']) == 7934 and len(linked['group_sizes']) == 3644
            and len(linked['cad_ids']) == 8046, 'full official counts differ')
    run = data_root / 'experiments/runs/idea01_text2cad_g4_full_20261002'
    entries = manifest_entries(run / 'SHA256SUMS')
    require(sha256(run / 'final_ledger.json') == certificate['final_ledger_sha256'], 'final score ledger differs')
    expected_chunks = {f'card{s}/chunk_s{s:02d}_t{t:05d}_{t+1:05d}.jsonl' for s in (0, 1)
                       for t in range(s*1822, (s+1)*1822)}
    require({n for n in entries if '/chunk_' in n} == expected_chunks, 'dense full matrix paths differ')
    new_output(output)
    started = time.time()
    write_json(output / 'startup.json', dict(pid=os.getpid(), argv=sys.argv, started_epoch=started,
               approved_sha=approved_sha, authorization_ref=authorization_ref, actual_head=git(repo, 'rev-parse', 'HEAD'),
               environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
               config_sha256=sha256(config_path), certificate_sha256=CERTIFICATE_SHA,
               source_module_sha256=sha256(Path(__file__)), zero_npu=True, score_forwards=0))
    write_json(output / 'input_fingerprints.json', fingerprints)
    groups, positives, sizes, lengths = (linked[k] for k in ('query_groups', 'positive_cads', 'group_sizes', 'lengths'))
    ranks = np.empty((7934, len(METHODS)), dtype=np.int32)
    asc, desc = length_ranks(lengths)
    ranks[:, 2] = asc[positives]; ranks[:, 3] = desc[positives]
    query_indices = [np.flatnonzero(groups == t) for t in range(3644)]
    rng = np.random.default_rng(SEED)
    chunk_fingerprints = {}
    with (output / 'text_top20.jsonl').open('x', encoding='utf-8', newline='\n') as top:
        for t in range(3644):
            shard = t // 1822; name = f'card{shard}/chunk_s{shard:02d}_t{t:05d}_{t+1:05d}.jsonl'
            total, mean, nbytes = load_chunk(run/name, entries[name], t, 8046,
                                            config['inputs']['pair_manifest']['sha256'], config['checkpoint_sha256'], lengths)
            sr, mr = score_ranks(total), score_ranks(mean)
            qi = query_indices[t]
            ranks[qi, 0] = sr[positives[qi]]; ranks[qi, 1] = mr[positives[qi]]
            permutation = rng.permutation(8046); random_ranks = np.empty(8046, dtype=np.int32)
            random_ranks[permutation] = np.arange(1, 8047)
            ranks[qi, 4] = random_ranks[positives[qi]]
            record = dict(text_index=t, n_original_queries=int(sizes[t]))
            for method, values, rr in (('sum', total, sr), ('mean', mean, mr)):
                order = np.argsort(rr)[:20]
                record[method] = [dict(rank=int(rr[ci]), cad_index=int(ci), cad_id=linked['cad_ids'][ci],
                                       score=float(values[ci])) for ci in order]
            top.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            chunk_fingerprints[name] = dict(sha256=entries[name], bytes=nbytes, rows=8046)
            if (t+1) % 100 == 0 or t == 3643:
                progress = dict(phase='ranking', completed_texts=t+1, total_texts=3644, raw_rows_read=(t+1)*8046,
                                seconds=time.time()-started, zero_npu=True)
                write_json(output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    write_json(output/'raw_chunk_fingerprints.json', chunk_fingerprints)
    with (output/'query_ranks.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for i, qid in enumerate(linked['query_ids']):
            record = dict(query_id=qid, text_index=int(groups[i]), cad_index=int(positives[i]),
                          cad_id=linked['cad_ids'][positives[i]], group_size=int(sizes[groups[i]]),
                          n_valid_total=int(lengths[positives[i]]),
                          ranks={method: int(ranks[i, j]) for j, method in enumerate(METHODS)})
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    write_json(output/'progress.json', dict(phase='cluster_bootstrap', replicates=REPLICATES))
    metrics = {method: rank_metrics(ranks[:, j]) for j, method in enumerate(METHODS)}
    results = dict(schema='text2cad-g5-cpu-evaluation-v1', run_id='idea01_text2cad_g5_cpu_20261003',
                   n_queries=7934, n_text_groups=3644, n_gallery=8046, raw_rows_read=29319624,
                   primary='sum', diagnostic='mean', tie_rule='exact score then cad_index ascending',
                   metric_units='fraction for recall, one-based rank', metrics=metrics,
                   random_analytic=dict(**{f'R@{k}': k/8046 for k in KS}, expected_rank=4023.5),
                   duplicate_uid_ceilings=duplicate_ceilings(sizes),
                   statistics=clustered_intervals(ranks, groups),
                   stratification=stratified_metrics(ranks, groups, sizes, positives, lengths),
                   sum_over_analytic_random={f'R@{k}': metrics['sum'][f'R@{k}']/(k/8046) for k in KS},
                   source_score_sha=SCORE_SHA, source_manifest_sha256=MANIFEST_SHA,
                   matrix_certificate_sha256=CERTIFICATE_SHA,
                   interpretation_scope='Protocol A official UID retrieval; label-smoothed native two-head model, not pure MLE/geometry NLL',
                   elapsed_seconds=time.time()-started, zero_npu=True, score_forwards=0)
    write_json(output/'metrics.json', results)
    write_json(output/'completion.json', dict(status='complete', completed_epoch=time.time(), raw_rows_read=29319624,
               query_ranks=7934, text_top20=3644, approved_sha=approved_sha, zero_npu=True))
    print(json.dumps(dict(status='complete', output=str(output), seconds=time.time()-started)), flush=True)
    return results
