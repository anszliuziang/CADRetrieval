# Research Findings — Text2CAD 2026-10-03

## Tested scope

Frozen original Text2CAD v1.0 native two-head SUM; full3644×8046 exact matrix, all7934 preeligible nonempty L0 original query UIDs. One evaluation, no result-driven method changes.

## Verdict and supported claim

Broad text-to-CAD useful/semantic retrieval claim: **partial**. Narrow fixed-protocol UID ranking signal claim: **yes, provisionally**. Fresh reviewer gpt-5.6-sol ultra, review_independence=same-family, acceptance_status=provisional. Root specialized numerical/original-byte audits all passed; this does not upgrade semantic independence.

SUM R@1/5/10/20=0.201664/0.920091/1.638518/2.609024%, MedR3411.5. All12 preregistered paired Top-K CI lower bounds against fixed random/two length orders>0. Absolute retrieval weak; whole mean score remains diagnostic. Shortest vs longest target R@20=4.918% vs0.306%, MedR551.5 vs6798. Unique-text MedR4944 shows aggregate gains are not uniform across rank statistics/strata.

## Gaps and constraints

Exact UID is not semantic/equivalence qrels; no shuffled/null/candidate-only control isolates text causality. Two simple length baselines do not eliminate other candidate priors. One checkpoint/dataset and exact-UID split does not establish design-family disjointness, general frozen-generator utility, length robustness or SOTA. IMMA different-gallery figures cannot be a reproduced comparator. Native two-head training includes label smoothing: do not call this pure MLE or geometry NLL. Bootstrap concerns sampled text groups, not training/model/random realization variance. Duplicate-text UID ceilings substantially exceed observed scores, so duplicates alone do not explain low recall.

## Reusable evidence and engineering lessons

G4all29319624 rows/117289444 checks, source/local/OBS3666 fullSHA, final matrix certificate passed. G5all39670 method ranks/145760 Top20/5000 actual draws independently recomputed:873364 checks allpassed. Full70payload OBS archive SHA and independent all-byte coverage actual0.

OBS implicit concurrent parent creation reproduced ENOENT; explicit precreation passed. Preserve real peer exits and full manifests, never infer success from a pipeline tail or filename count. Review pytest fixture is distinct from declared proof: pin exact proof manifest, copy explicit files, record known excluded fixture, reject other extras. Failures retained and no scoring repeated.

## Optional future research, not executed

Pre-register query permutation/null/candidate prior controls; same-gallery retrieval baseline; within-length or length-matched analysis; blind multi-positive semantics; application threshold and separate checkpoint/data/family replication. No additional experiment launched in this authorization.

## Provenance

Full table and limitations: submissions/idea01_text2cad_benchmark_20260928/CODEX_TEXT2CAD_FULL_RETRIEVAL_RESULTS_2026-10-03.md. Numeric/byte certificate: reviews/IDEA01_M1C_G5_EVALUATION_CERTIFICATE_2026-10-03.json. Original prompt/raw verdict/normalized routing in .aris/traces/result-to-claim/2026-10-03_run01 (full ignored trace archived as evidence). Generic evidence_check.py unavailable; recorded honestly, not fabricated. Specialized project audits provide actual experiment integrity evidence.
