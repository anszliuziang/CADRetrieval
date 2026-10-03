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
REMOTE = '/data/obs_data_liuza/CADRetrieval/runs/' + RUN_ID + '_recovery_v2_20261003'
DIAG_REMOTE = '/data/obs_data_liuza/CADRetrieval/runs/m1c_g4_obs_receiver_diag_20261003'
REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / 'experiments/runs' / RUN_ID
CONTROL = REPO.parent / (REPO.name + '_runtime') / 'g4_archive_recovery_v2_20261003'
DIAG_CONTROL = REPO.parent / (REPO.name + '_runtime') / 'g4_obs_receiver_diag_20261003'


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


def destination_commands(remote, entries, precreate):
    if not PurePosixPath(remote).is_absolute() or '..' in PurePosixPath(remote).parts:
        raise ValueError('destination must be a safe absolute POSIX path')
    parents = set()
    for name, _, _ in entries:
        parts = PurePosixPath(name).parts
        if (not parts or '..' in parts or PurePosixPath(name).is_absolute() or
                '\\' in name or ':' in name or name.startswith('-') or name != '/'.join(parts)):
            raise ValueError('unsafe payload path for directory preparation')
        for i in range(1, len(parts)):
            parents.add('/'.join(parts[:i]))
    commands = ['mkdir -- ' + shlex.quote(remote)]
    if precreate and parents:
        paths = [shlex.quote(remote + '/' + p) for p in sorted(parents)]
        commands.extend(['mkdir -p -- ' + ' '.join(paths),
                         ' && '.join('test -d ' + p for p in paths)])
    return commands


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


def logged_command(argv, control, label, check=True):
    write_json(control / (label + '_command.json'), dict(argv=argv, started=time.time()))
    with (control / (label + '_stdout.log')).open('wb') as out, \
            (control / (label + '_stderr.log')).open('wb') as err:
        result = subprocess.run(argv, stdout=out, stderr=err)
    write_json(control / (label + '_exit.json'), dict(exit_code=result.returncode, ended=time.time()))
    if check and result.returncode:
        raise RuntimeError(label + ' failed; see durable logs')
    return result.returncode


def verify_code_authorization(approved, mode):
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
            '`gpu_budget`: 0' not in active or '`max_scored_pairs`: 0' not in active or
            f'`archive_mode`: `{mode}`' not in active):
        raise ValueError('ACTIVE_TASK missing exact zero-NPU transport authorization')
    return dict(approved_sha=approved, authorized_mode=mode, actual_head=git('rev-parse', 'HEAD').decode().strip(),
                script_sha256=digest(Path(__file__)), active_sha256=digest(REPO / 'experiments/ACTIVE_TASK.md'))


def transport(source, entries, remote, control, precreate):
    # This routine is shared by the real OBS diagnostic and the full archive.
    for i, command in enumerate(destination_commands(remote, entries, precreate)):
        logged_command(ssh(command), control, 'prepare' + str(i))
    tar = shutil.which('tar')
    if not tar:
        raise RuntimeError('system tar unavailable')
    pipelines = []
    for index, group in enumerate(partition(entries, 8)):
        names = [e[0] for e in group] + (['SHA256SUMS'] if index == 0 else [])
        listing = control / ('stream' + str(index) + '.txt')
        listing.write_bytes(('\n'.join(names) + '\n').encode('utf-8'))
        sender = [tar, '-czf', '-', '-C', str(source), '-T', str(listing)]
        receiver = ssh('tar -xzf - --no-same-owner -C ' + shlex.quote(remote))
        pipelines.append((sender, receiver, control, 'stream' + str(index)))
    write_json(control / 'transport_started.json', dict(streams=len(pipelines), precreate=precreate,
               remote=remote, payload_files=len(entries), started=time.time()))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_pipeline, *p) for p in pipelines]
        exits = [f.result() for f in futures]
    result = dict(pipelines=exits, precreate=precreate, ok=all(e['ok'] for e in exits))
    if result['ok']:
        result['sha256_exit'] = logged_command(
            ssh('cd ' + shlex.quote(remote) + ' && sha256sum -c SHA256SUMS'),
            control, 'remote_sha256', check=False)
        result['ok'] = result['sha256_exit'] == 0
    write_json(control / 'transport_result.json', result)
    return result


def diagnostic_fixture(root):
    root.mkdir(exist_ok=False)
    lines = []
    for i in range(16):
        parent = 'card0' if i < 8 else ('card1' if i < 15 else 'outer_launch/deep')
        name = parent + '/file%02d.bin' % i
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = hashlib.sha256(('OBS-diagnostic-%02d' % i).encode()).digest() * 2048
        path.write_bytes(payload)
        lines.append(digest(path) + '  ./' + name + '\n')
    (root / 'SHA256SUMS').write_bytes(''.join(lines).encode())
    return read_manifest(root, digest(root / 'SHA256SUMS'), 16, 16 * 65536)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--approved-sha', required=True)
    ap.add_argument('--mode', choices=('diagnostic', 'full'), required=True)
    args = ap.parse_args()
    provenance = verify_code_authorization(args.approved_sha, args.mode)
    control = DIAG_CONTROL if args.mode == 'diagnostic' else CONTROL
    create_control(control)
    result = dict(status='running', mode=args.mode, phase='source_verify', pid=os.getpid(),
                  started=time.time(), provenance=provenance, zero_npu=True)
    try:
        write_json(control / 'state.json', result)
        if args.mode == 'diagnostic':
            source = control / 'source'
            entries = diagnostic_fixture(source)
            if 2 * (sum(e[2] for e in entries) + (source / 'SHA256SUMS').stat().st_size) > 16 * 1024 * 1024:
                raise ValueError('diagnostic byte budget exceeded')
            logged_command(ssh('mkdir -- ' + shlex.quote(DIAG_REMOTE)), control, 'diagnostic_root')
            result.update(phase='diagnostic', manifest_sha256=digest(source / 'SHA256SUMS'),
                          payload_files=16, payload_bytes_per_condition=sum(e[2] for e in entries))
            write_json(control / 'state.json', result)
            conditions = {}
            for label, precreate in (('implicit_parents', False), ('prepared_parents', True)):
                sub = control / label
                create_control(sub)
                conditions[label] = transport(source, entries, DIAG_REMOTE + '/' + label, sub, precreate)
                write_json(control / 'conditions.json', conditions)
            result.update(status='diagnostic_complete', conditions=conditions,
                          prepared_passed=conditions['prepared_parents']['ok'])
        else:
            # A full run is permitted only after the real OBS prepared diagnostic.
            diagnostic = json.loads((DIAG_CONTROL / 'state.json').read_text(encoding='utf-8'))
            if diagnostic.get('status') != 'diagnostic_complete' or diagnostic.get('prepared_passed') is not True:
                raise ValueError('real OBS prepared diagnostic did not pass')
            if diagnostic['provenance']['approved_sha'] != args.approved_sha:
                raise ValueError('diagnostic used different approved code')
            entries = read_manifest(SOURCE, MANIFEST_SHA, 3666, 14858752230)
            result.update(phase='transport', source=str(SOURCE), destination='cvteam14:' + REMOTE,
                          manifest_sha256=MANIFEST_SHA, verified_source_files=len(entries))
            write_json(control / 'state.json', result)
            transfer = transport(SOURCE, entries, REMOTE, control, True)
            result['transfer'] = transfer
            if not transfer['ok']:
                raise RuntimeError('transport failed; preserve evidence before any root-cause-guided recovery')
            result.update(status='transport_and_sha_passed', phase='await_independent_obs_audit')
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    result['ended'] = time.time()
    write_json(control / 'state.json', result)
    ok = result['status'] == 'transport_and_sha_passed' or (
        result['status'] == 'diagnostic_complete' and result['prepared_passed'])
    code = 0 if ok else 1
    write_json(control / 'controller_exit.json', dict(exit_code=code, status=result['status'], ended=time.time()))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
