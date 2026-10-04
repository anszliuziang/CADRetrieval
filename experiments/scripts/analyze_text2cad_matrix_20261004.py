"""Exploratory text alignment diagnostics on the frozen G4 matrix; no forwards."""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

KEYS = ('R@1', 'R@5', 'R@10', 'R@20', 'MedR', 'MeanR')
SEED, REPLICATES = 20261004, 5000


def require(ok, why):
    if not ok: raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                          encoding='utf-8', newline='\n')


def source_pin(path, canonical_pin):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    canonical = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    require(canonical == canonical_pin, 'Python source content changed')
    return dict(actual_sha256=actual, canonical_lf_sha256=canonical)


def rank_row(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size and np.isfinite(a).all(), 'invalid scores')
    out = np.empty(a.size, dtype=np.int32)
    out[np.lexsort((np.arange(a.size), -a))] = np.arange(1, a.size+1)
    return out


def metrics(ranks):
    a = np.asarray(ranks)
    require(a.ndim == 1 and len(a) and np.isfinite(a).all() and (a >= 1).all(), 'invalid ranks')
    return {**{f'R@{k}': float(np.mean(a <= k)) for k in (1, 5, 10, 20)},
            'MedR': float(np.median(a)), 'MeanR': float(np.mean(a))}


def assigned_ranks(ranks, groups, positives, permutation):
    return ranks[np.asarray(permutation)[groups], positives]


def wrong_expectation(ranks, groups, positives):
    groups, positives = np.asarray(groups), np.asarray(positives)
    g = len(ranks); require(g > 1 and len(groups) > 0, 'no wrong rows')
    own = ranks[groups, positives]
    result = {f'R@{k}': float(np.mean(((ranks <= k).sum(axis=0)[positives] - (own <= k))/(g-1)))
              for k in (1, 5, 10, 20)}
    result['MeanR'] = float(np.mean((ranks.sum(axis=0, dtype=np.int64)[positives]-own)/(g-1)))
    # Medians cannot be averaged to obtain a population median; intentionally omitted.
    return result


def decomposition(scores):
    a = np.asarray(scores); g, n = a.shape
    grand = a.mean(); row = a.mean(axis=1); col = a.mean(axis=0)
    row_ss = float(n*np.square(row-grand).sum())
    col_ss = float(g*np.square(col-grand).sum())
    interaction = sum(float(np.square(a[t]-row[t]-col+grand).sum()) for t in range(g))
    total = row_ss+col_ss+interaction
    return dict(row_ss=row_ss, column_ss=col_ss, interaction_ss=interaction,
                row_fraction=row_ss/total if total else 0.,
                column_fraction=col_ss/total if total else 0.,
                interaction_fraction=interaction/total if total else 0.,
                column_fraction_after_row_offset=col_ss/(col_ss+interaction) if col_ss+interaction else 0.)


def candidate_diagnostics(scores, ranks, sizes, k=20):
    g, n = scores.shape; k = min(k, n); sizes = np.asarray(sizes)
    prior = rank_row(scores.mean(axis=0)); prior_top = prior <= k
    overlap = np.asarray([np.count_nonzero((r <= k) & prior_top)/k for r in ranks])
    rho = np.asarray([1-6*np.square(r.astype(float)-prior).sum()/(n*(n*n-1)) for r in ranks])
    counts = (ranks <= k).sum(axis=0)
    order = np.lexsort((np.arange(n), -counts))[:20]
    return dict(prior_ranks=prior.tolist(), k=k,
                topk_overlap_group_mean=float(overlap.mean()),
                topk_overlap_query_weighted=float(np.average(overlap, weights=sizes)),
                topk_overlap_group_quantiles=np.quantile(overlap, [.05,.5,.95]).tolist(),
                spearman_group_mean=float(rho.mean()), spearman_query_weighted=float(np.average(rho, weights=sizes)),
                spearman_group_quantiles=np.quantile(rho, [.05,.5,.95]).tolist(),
                topk_unique_candidates=int(np.count_nonzero(counts)),
                most_frequent_topk=[dict(cad_index=int(i), text_groups=int(counts[i]),
                                         group_fraction=float(counts[i]/g)) for i in order])


def selectivity(scores, groups, positives):
    # Query weights are preserved by including each original query exactly once.
    wins, ties = [], []
    for t, c in zip(groups, positives):
        col = scores[:, c]; own = col[t]
        wins.append(float(np.count_nonzero(col < own)/(len(col)-1)))
        ties.append(float((np.count_nonzero(col == own)-1)/(len(col)-1)))
    return dict(strict_win_fraction=wins, tie_fraction=ties)


def permutation_metrics(rank_matrices, groups, positives, draws):
    out = np.empty((len(rank_matrices), len(draws), len(KEYS)))
    for j, permutation in enumerate(draws):
        for a, ranks in enumerate(rank_matrices):
            m = metrics(assigned_ranks(ranks, groups, positives, permutation))
            out[a, j] = [m[key] for key in KEYS]
    return out


def run(args):
    repo, rt, out = Path(args.repo).resolve(), Path(args.runtime).resolve(), Path(args.output).resolve()
    git = lambda *a: subprocess.check_output(['git', '-C', str(repo), *a], text=True).strip()
    require(git('rev-parse', 'HEAD') == args.approved_sha, 'unapproved HEAD')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'dirty tracked files')
    helper = repo/'experiments/src/text2cad_g5_cpu.py'
    helper_pin = source_pin(helper, 'de76f9057b8759d6f02c18ac1f2c9d143f4f639d42d2e461378870a3ade5e373')
    spec = importlib.util.spec_from_file_location('g5_read_only', helper)
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    cp = repo/'experiments/configs/text2cad_g5_cpu_frozen_20261003.json'
    require(digest(cp) == 'f75356ae738f40faf756747e6c61aec5ad38d6696cfda34a47dcf04612737b77', 'G5 pins differ')
    cfg = e.read_json(cp); cert = e.require_certificate(repo/cfg['certificate_path'], cfg['certificate_sha256'])
    pins = {k: (repo if p['root']=='data' else rt)/p['path'] for k,p in cfg['inputs'].items()}
    for k,p in pins.items(): require(digest(p) == cfg['inputs'][k]['sha256'], 'input changed '+k)
    linked = e.link_inputs(**{k: e.jsonl(pins[k]) for k in ('queries','gallery','qrels','texts','store','lengths')},
                           mapping=e.read_json(pins['mapping']))
    groups, positives, sizes, lengths = (linked[k] for k in ('query_groups','positive_cads','group_sizes','lengths'))
    require((len(sizes), len(lengths), len(groups)) == (3644,8046,7934), 'incomplete matrix')
    raw = repo/'experiments/runs/idea01_text2cad_g4_full_20261002'; entries = e.manifest_entries(raw/'SHA256SUMS')
    require(digest(raw/'final_ledger.json') == cert['final_ledger_sha256'], 'score ledger differs')
    formal = repo/'experiments/runs/idea01_text2cad_g5_cpu_20261003'
    require(digest(formal/'metrics.json') == '6005aa72961c06cf4cf7d1fa9fc59f81e2e31f63b18fe8d94a9d8b6d60d3dec5', 'G5 metrics differ')
    # Bind every original positive rank to the certified formal artifact, not a report.
    fm = e.manifest_entries(raw/'SHA256SUMS')
    formal_manifest = (formal/'SHA256SUMS').read_text()
    require(hashlib.sha256(formal_manifest.encode()).hexdigest() == 'c98e205626a9d95d4830f3c628518f5f5e34f215c80b2079948dc0e54b19be95', 'G5 manifest differs')
    expected_query_sha = dict((name,sha) for sha,name in (line.split('  ',1) for line in formal_manifest.splitlines()))['query_ranks.jsonl']
    require(digest(formal/'query_ranks.jsonl') == expected_query_sha, 'G5 positive ranks differ')
    old = {r['query_id']: r for r in e.jsonl(formal/'query_ranks.jsonl')}
    require(set(old) == set(linked['query_ids']) and len(old) == 7934, 'formal rank coverage')
    out.mkdir(parents=True, exist_ok=False); started = time.time()
    dump(out/'startup.json', dict(pid=os.getpid(), argv=sys.argv, approved_sha=args.approved_sha,
         code_sha256=digest(Path(__file__)), helper_source=helper_pin, python=sys.version, numpy=np.__version__, platform=platform.platform(),
         seed=SEED, replicates=REPLICATES, exploratory=True, zero_npu=True, score_forwards=0,
         inputs={k:dict(path=str(p),sha256=digest(p)) for k,p in pins.items()},
         raw_manifest_sha256=digest(raw/'SHA256SUMS'), formal_manifest_sha256=digest(formal/'SHA256SUMS')))
    scores = np.empty((2,3644,8046), dtype=np.float64)
    ranks = np.empty(scores.shape, dtype=np.int32)
    chunks = {}
    for t in range(3644):
        s=t//1822; name=f'card{s}/chunk_s{s:02d}_t{t:05d}_{t+1:05d}.jsonl'
        total, mean, b = e.load_chunk(raw/name, fm[name], t,8046,cfg['inputs']['pair_manifest']['sha256'],cfg['checkpoint_sha256'],lengths)
        for a,v in enumerate((total,mean)): scores[a,t]=v; ranks[a,t]=rank_row(v)
        chunks[name]=dict(sha256=entries[name],bytes=b)
        if (t+1)%100 == 0 or t==3643:
            dump(out/'progress.json', dict(phase='extracting_existing_scores',texts=t+1,seconds=time.time()-started))
            print('extracted',t+1,flush=True)
    actual = [ranks[a,groups,positives] for a in range(2)]
    for i,q in enumerate(linked['query_ids']):
        require(old[q]['text_index']==int(groups[i]) and old[q]['cad_index']==int(positives[i])
                and all(old[q]['ranks'][m]==int(actual[a][i]) for a,m in enumerate(('sum','mean'))), 'G5 rank mismatch '+q)
    dump(out/'chunk_fingerprints.json',chunks)
    np.save(out/'rank_matrices.npy',ranks,allow_pickle=False)
    rng=np.random.default_rng(SEED)
    draws=np.stack([rng.permutation(3644).astype(np.int32) for _ in range(REPLICATES)])
    np.save(out/'permutations.npy',draws,allow_pickle=False)
    dump(out/'progress.json',dict(phase='permutation_and_candidate_diagnostics',seconds=time.time()-started))
    null=permutation_metrics(ranks,groups,positives,draws)
    np.save(out/'permutation_metrics.npy',null,allow_pickle=False)
    results=dict(exploratory=True,n_queries=7934,n_text_groups=3644,n_gallery=8046,
                 restored_formal_positive_ranks=15868,seed=SEED,replicates=REPLICATES,arms={})
    qtiles=np.quantile(lengths,[.25,.5,.75]); require(qtiles.tolist()==[27.,40.,70.], 'fixed length strata differs')
    layers={'unique':sizes[groups]==1,'repeated':sizes[groups]>1}
    bounds=np.searchsorted(qtiles,lengths[positives],side='left')
    layers.update({f'length_quartile_{i+1}':bounds==i for i in range(4)})
    records=[dict(query_id=q,text_index=int(groups[i]),cad_index=int(positives[i])) for i,q in enumerate(linked['query_ids'])]
    for a,method in enumerate(('sum','mean')):
        observed=metrics(actual[a]); candidate=candidate_diagnostics(scores[a],ranks[a],sizes)
        prior=np.asarray(candidate.pop('prior_ranks'))
        np.save(out/f'{method}_empirical_prior_ranks.npy',prior,allow_pickle=False)
        candidate['fixed_ranking_uid_metrics']=metrics(prior[positives])
        candidate['length_score_correlation']=float(np.corrcoef(lengths,scores[a].mean(axis=0))[0,1])
        sel=selectivity(scores[a],groups,positives)
        for i,r in enumerate(records): r[method]=dict(rank=int(actual[a][i]),strict_score_win=sel['strict_win_fraction'][i],score_tie=sel['tie_fraction'][i])
        results['arms'][method]=dict(observed=observed,wrong_text_exact_expectation=wrong_expectation(ranks[a],groups,positives),
             permutation={key:dict(mean=float(null[a,:,j].mean()),quantiles_2_5_50_97_5=np.quantile(null[a,:,j],[.025,.5,.975]).tolist(),
                           one_sided_randomization_p=float((1+np.count_nonzero(null[a,:,j]>=observed[key] if j<4 else null[a,:,j]<=observed[key]))/(REPLICATES+1))) for j,key in enumerate(KEYS)},
             candidate=candidate,score_variance=decomposition(scores[a]),
             positive_score_selectivity=dict(strict_win_query_mean=float(np.mean(sel['strict_win_fraction'])),tie_query_mean=float(np.mean(sel['tie_fraction'])),
                                            win_query_quantiles=np.quantile(sel['strict_win_fraction'],[.05,.5,.95]).tolist()),
             strata={key:dict(n_queries=int(mask.sum()),observed=metrics(actual[a][mask]),
                              wrong_text_exact_expectation=wrong_expectation(ranks[a],groups[mask],positives[mask])) for key,mask in layers.items()})
    with (out/'query_diagnostics.jsonl').open('x',encoding='utf-8',newline='\n') as f:
        for row in records: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    results['elapsed_seconds']=time.time()-started
    dump(out/'results.json',results)
    require(git('rev-parse','HEAD')==args.approved_sha and not git('status','--porcelain','--untracked-files=no'), 'Git changed during analysis')
    dump(out/'completion.json',dict(status='ok',zero_npu=True,score_forwards=0,seconds=time.time()-started))
    files=sorted(p for p in out.iterdir() if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in files),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='ok',seconds=time.time()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','runtime','output','approved-sha'): p.add_argument('--'+name,required=True)
    args=p.parse_args()
    try: run(args)
    except Exception as exc:
        out=Path(args.output)
        # Failure evidence is written only in the output created by this invocation.
        if out.is_dir() and (out/'startup.json').is_file():
            s=json.loads((out/'startup.json').read_text(encoding='utf-8'))
            if s.get('pid')==os.getpid(): dump(out/'failure.json',dict(error=type(exc).__name__,message=str(exc)))
        raise
