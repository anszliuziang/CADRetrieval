"""Small final report/control supplement; reuse approved zero-NPU transport."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
RT = REPO.parent / (REPO.name + '_runtime')
SOURCE = RT / 'g5_final_delivery_proof_bundle_20261003'
CONTROL = RT / 'g5_final_delivery_proof_control_20261003'
TARGET = '/data/obs_data_liuza/CADRetrieval/runs/m1c_g5_final_delivery_proof_20261003'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def copy_file(src, dst):
    if not src.is_file() or src.is_symlink():
        raise ValueError('missing/symlink payload: ' + str(src))
    before = digest(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if digest(dst) != before:
        raise ValueError('copy changed bytes: ' + str(src))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--approved-sha', required=True)
    args = ap.parse_args()
    git = lambda *a: subprocess.check_output(['git', '-C', str(REPO), *a])
    rel = Path(__file__).relative_to(REPO).as_posix()
    if git('show', args.approved_sha + ':' + rel) != Path(__file__).read_bytes().replace(b'\r\n', b'\n'):
        raise ValueError('supplement script differs from approved Git bytes')
    active = (REPO / 'experiments/ACTIVE_TASK.md').read_text(encoding='utf-8')
    if (f'`approved_finalproof_sha`: `{args.approved_sha}`' not in active or
            '`finalproof_transport_authorized`: true' not in active or
            '`gpu_budget`: 0' not in active or '`max_scored_pairs`: 0' not in active):
        raise ValueError('missing zero-NPU final proof authorization')
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise ValueError('dirty tracked checkout')
    base_path = REPO / 'experiments/scripts/run_g4_archive_recovery.py'
    base_sha = '8bc4eeabea57f416982a286add9b65ba35f57ed1'
    if git('show', base_sha + ':experiments/scripts/run_g4_archive_recovery.py') != base_path.read_bytes().replace(b'\r\n', b'\n'):
        raise ValueError('reused approved transport differs')
    spec = importlib.util.spec_from_file_location('reviewed_transport', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)  # main() is never called; generic audited transport only.
    SOURCE.mkdir(exist_ok=False)
    CONTROL.mkdir(exist_ok=False)
    save = lambda n, o: base.write_json(CONTROL / n, o)
    try:
        files = [
            'experiments/ACTIVE_TASK.md', 'experiments/findings.md',
            'experiments/reviews/IDEA01_M1C_G5_EVALUATION_CERTIFICATE_2026-10-03.json',
            'experiments/reviews/IDEA01_M1C_G5_CODEX_FINAL_RESULTS_REVIEW_2026-10-03.md',
            'experiments/reviews/IDEA01_M1C_G5_CLAIMS_FROM_RESULTS_2026-10-03.md',
            'experiments/reviews/IDEA01_M1C_G5_ARCHIVE_APPROVAL_2026-10-03.md',
            'experiments/submissions/idea01_text2cad_benchmark_20260928/CODEX_TEXT2CAD_FULL_RETRIEVAL_RESULTS_2026-10-03.md',
            'experiments/scripts/archive_text2cad_final_proof_20261003.py',
            'experiments/scripts/archive_text2cad_g5_results_20261003.py',
            'experiments/scripts/run_g4_archive_recovery.py',
            'experiments/tests/test_g5_archive_operator_boundaries_20261003.py',
        ]
        for name in files:
            copy_file(REPO / name, SOURCE / 'project' / name)
        for label, directory in [
            ('archive_v2_control', RT / 'g5_complete_archive_control_v2_20261003'),
            ('archive_first_local_failure', RT / 'g5_complete_archive_control_20261003'),
            ('semantic_trace', REPO / '.aris/traces/result-to-claim/2026-10-03_run01'),
        ]:
            for p in sorted(directory.iterdir()):
                copy_file(p, SOURCE / label / p.name)
        for name in [
            'g5_complete_archive_launcher_20261003.json',
            'g5_complete_archive_stdout_20261003.log', 'g5_complete_archive_stderr_20261003.log',
            'g5_complete_archive_v2_launcher_20261003.json',
            'g5_complete_archive_v2_stdout_20261003.log', 'g5_complete_archive_v2_stderr_20261003.log',
            'g5_archive_boundary_tests_20261003.json', 'g5_archive_review_scope_tests_20261003.json',
            'g5_final_review_responses_20261003.json', 'g5_managed_worktree_archive_20261003.json',
        ]:
            copy_file(RT / name, SOURCE / 'operator' / name)
        entries = {p.relative_to(SOURCE).as_posix(): digest(p) for p in SOURCE.rglob('*') if p.is_file()}
        raw = ''.join(h + '  ' + n + '\n' for n, h in sorted(entries.items())).encode()
        (SOURCE / 'SHA256SUMS').write_bytes(raw)
        mh = digest(SOURCE / 'SHA256SUMS')
        total = sum((SOURCE / n).stat().st_size for n in entries)
        payload = base.read_manifest(SOURCE, mh, len(entries), total)
        save('source.json', dict(files=len(entries), bytes=total, manifest_sha256=mh,
             target=TARGET, approved_sha=args.approved_sha, reused_transport_sha=base_sha,
             pid=os.getpid(), started=time.time(), zero_npu=True))
        transfer = base.transport(SOURCE, payload, TARGET, CONTROL, True)
        if not transfer['ok']:
            raise RuntimeError('proof transport failed; retain all evidence')
        witness = (RT / 'g5_complete_archive_control_v2_20261003/independent_readonly_witness.py').read_bytes()
        if hashlib.sha256(witness).hexdigest() != 'a22d785a6f66d620d31fc13446ebff659273b0c7ac9a75e576aa16ea619902d3':
            raise ValueError('readonly witness pin')
        argv = base.ssh('python3 -B - ' + TARGET + ' ' + mh)
        save('independent_command.json', dict(argv=argv, witness_sha256=hashlib.sha256(witness).hexdigest()))
        result = subprocess.run(argv, input=witness, capture_output=True)
        (CONTROL / 'independent_stdout.json').write_bytes(result.stdout)
        (CONTROL / 'independent_stderr.log').write_bytes(result.stderr)
        save('independent_exit.json', dict(actual_exit_code=result.returncode))
        if result.returncode:
            raise RuntimeError('independent readonly proof byte audit failed')
        observed = json.loads(result.stdout)
        if (observed['all_passed'] is not True or observed['verified_files'] != len(entries) or
                observed['payload_bytes'] != total or observed['manifest_sha256'] != mh or
                observed['unknown_extra'] != []):
            raise ValueError('independent proof coverage/count differs')
        save('completion.json', observed)
        save('controller_exit.json', dict(actual_exit_code=0, pid=os.getpid(), ended=time.time()))
        print(json.dumps(observed), flush=True)
        return 0
    except BaseException as exc:
        save('controller_exit.json', dict(actual_exit_code=1, pid=os.getpid(), error=repr(exc), ended=time.time()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
