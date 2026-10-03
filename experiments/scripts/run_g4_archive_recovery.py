"""CPU-only, manifest-pinned G4 transport. Run once as a detached local job.

Only system tar and sha256sum run remotely; no project source is copied there.
Existing destination/control directories are never reused or removed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import time

MANIFEST_SHA = '97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733'
RUN_ID = 'idea01_text2cad_g4_full_20261002'
REMOTE = '/data/obs_data_liuza/CADRetrieval/runs/' + RUN_ID + '_recovery_20261003'
REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'experiments/runs' / RUN_ID
CONTROL = REPO.parent / (REPO.name + '_runtime') / 'g4_archive_recovery_20261003'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_manifest(root, expected_sha, expected_count, expected_bytes):
    root = root.resolve()
    manifest = root / 'SHA256SUMS'
    if manifest.is_symlink():
        raise ValueError('symlink manifest')
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha or b'\r' in raw or not raw.endswith(b'\n'):
        raise ValueError('manifest pin/LF mismatch')
    entries = []
    names = set()
    for line in raw.decode('utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise ValueError('malformed manifest')
        expected, name = match.groups()
        if name.startswith('./'):
            name = name[2:]
        parts = PurePosixPath(name).parts
        if (not parts or PurePosixPath(name).is_absolute() or '..' in parts or
                '\\' in name or ':' in name or name.startswith('-') or name in names or
                name == 'SHA256SUMS' or name != '/'.join(parts)):
            raise ValueError('unsafe/duplicate/self path: ' + name)
        names.add(name)
        path = root / name
        if (root not in path.resolve().parents or not path.is_file() or
                any(root.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1))):
            raise ValueError('missing/symlink/outside path: ' + name)
        if digest(path) != expected:
            raise ValueError('source byte mismatch: ' + name)
        entries.append((name, expected, path.stat().st_size))
    if len(entries) != expected_count or sum(e[2] for e in entries) != expected_bytes:
        raise ValueError('source count/size mismatch')
    return entries


def partition(entries, streams):
    if not entries or not 1 <= streams <= 8:
        raise ValueError('invalid stream count')
    groups = [[] for _ in range(min(len(entries), streams))]
    sizes = [0] * len(groups)
    for entry in sorted(entries, key=lambda e: (-e[2], e[0])):
        index = min(range(len(groups)), key=lambda i: (sizes[i], i))
        groups[index].append(entry)
        sizes[index] += entry[2]
    return groups


def create_control(path):
    path.mkdir(parents=True, exist_ok=False)


def run_pipeline(sender_argv, receiver_argv, control, label):
    started = time.time()
    sender = receiver = None
    errors = []
    launch = dict(sender_argv=sender_argv, receiver_argv=receiver_argv,
                  sender_pid=None, receiver_pid=None, started=started)
    try:
        write_json(control / (label + '_launch.json'), launch)
        with (control / (label + '_sender_stderr.log')).open('wb') as sender_log, \
                (control / (label + '_receiver_stderr.log')).open('wb') as receiver_log, \
                (control / (label + '_receiver_stdout.log')).open('wb') as receiver_out:
            sender = subprocess.Popen(sender_argv, stdout=subprocess.PIPE, stderr=sender_log)
            launch['sender_pid'] = sender.pid
            write_json(control / (label + '_launch.json'), launch)
            try:
                receiver = subprocess.Popen(receiver_argv, stdin=sender.stdout, stdout=receiver_out, stderr=receiver_log)
            finally:
                sender.stdout.close()
            launch['receiver_pid'] = receiver.pid
            write_json(control / (label + '_launch.json'), launch)
            receiver.wait()
            sender.wait()
    except BaseException as exc:
        errors.append(type(exc).__name__ + ': ' + str(exc))
        for process in (sender, receiver):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    result = dict(sender_exit=sender.returncode if sender else None,
                  receiver_exit=receiver.returncode if receiver else None,
                  sender_pid=sender.pid if sender else None,
                  receiver_pid=receiver.pid if receiver else None,
                  sender_argv=sender_argv, receiver_argv=receiver_argv,
                  ok=not errors and sender is not None and receiver is not None and
                  sender.returncode == receiver.returncode == 0,
                  errors=errors, started=started, ended=time.time())
    write_json(control / (label + '_exit.json'), result)
    return result


def ssh(command):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'cvteam14', command]


def logged_command(argv, control, label):
    write_json(control / (label + '_command.json'), dict(argv=argv, started=time.time()))
    with (control / (label + '_stdout.log')).open('wb') as out, \
            (control / (label + '_stderr.log')).open('wb') as err:
        result = subprocess.run(argv, stdout=out, stderr=err)
    write_json(control / (label + '_exit.json'), dict(exit_code=result.returncode, ended=time.time()))
    if result.returncode:
        raise RuntimeError(label + ' failed; see durable logs')


def verify_code_authorization(approved):
    if not re.fullmatch('[0-9a-f]{40}', approved):
        raise ValueError('approved SHA must be complete')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(REPO), *args])
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise ValueError('dirty tracked checkout')
    rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    if git('show', approved + ':' + rel) != Path(__file__).read_bytes():
        raise ValueError('actual script differs from approved Git blob')
    active = (REPO / 'experiments/ACTIVE_TASK.md').read_text(encoding='utf-8')
    if (f'`approved_sha`: `{approved}`' not in active or
            '`state`: `pilot_authorized`' not in active or
            '`gpu_budget`: 0' not in active or '`max_scored_pairs`: 0' not in active):
        raise ValueError('ACTIVE_TASK missing exact zero-NPU transport authorization')
    return dict(approved_sha=approved, actual_head=git('rev-parse', 'HEAD').decode().strip(),
                script_sha256=digest(Path(__file__)), active_sha256=digest(REPO / 'experiments/ACTIVE_TASK.md'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--approved-sha', required=True)
    args = ap.parse_args()
    provenance = verify_code_authorization(args.approved_sha)
    create_control(CONTROL)
    result = dict(status='running', phase='source_verify', pid=os.getpid(), started=time.time(),
                  source=str(SOURCE), destination='cvteam14:' + REMOTE, manifest_sha256=MANIFEST_SHA,
                  provenance=provenance, zero_npu=True)
    try:
        write_json(CONTROL / 'state.json', result)
        entries = read_manifest(SOURCE, MANIFEST_SHA, 3666, 14858752230)
        groups = partition(entries, 8)
        tar = shutil.which('tar')
        if not tar:
            raise RuntimeError('system tar unavailable')
        result.update(phase='create_destination', verified_source_files=len(entries))
        write_json(CONTROL / 'state.json', result)
        # Exclusive mkdir, no -p: an existing target fails without writes to it.
        logged_command(ssh('mkdir -- ' + shlex.quote(REMOTE)), CONTROL, 'mkdir')
        pipelines = []
        for index, group in enumerate(groups):
            names = [e[0] for e in group] + (['SHA256SUMS'] if index == 0 else [])
            listing = CONTROL / ('stream' + str(index) + '.txt')
            listing.write_bytes(('\n'.join(names) + '\n').encode('utf-8'))
            sender = [tar, '-czf', '-', '-C', str(SOURCE), '-T', str(listing)]
            receiver = ssh('tar -xzf - --no-same-owner -C ' + shlex.quote(REMOTE))
            pipelines.append((sender, receiver, CONTROL, 'stream' + str(index)))
        result.update(phase='transport', streams=len(pipelines))
        write_json(CONTROL / 'state.json', result)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(run_pipeline, *p) for p in pipelines]
            exits = [f.result() for f in futures]
        result['pipelines'] = exits
        if not all(e['ok'] for e in exits):
            raise RuntimeError('transport pipeline failed; no automatic retry')
        result.update(phase='remote_sha256')
        write_json(CONTROL / 'state.json', result)
        logged_command(ssh('cd ' + shlex.quote(REMOTE) + ' && sha256sum -c SHA256SUMS'), CONTROL, 'remote_sha256')
        result.update(status='transport_and_sha_passed', phase='await_independent_obs_audit')
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    result['ended'] = time.time()
    write_json(CONTROL / 'state.json', result)
    code = 0 if result['status'] == 'transport_and_sha_passed' else 1
    write_json(CONTROL / 'controller_exit.json', dict(exit_code=code, status=result['status'], ended=time.time()))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
