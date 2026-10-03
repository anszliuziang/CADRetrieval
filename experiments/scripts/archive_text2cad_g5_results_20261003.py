"""Archive explicit G5 frozen result/audit/code/input proofs with real exits."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

repo=Path('D:/projects/IdeaSpark/CADRetrievalIdea1');rt=repo.parent/'CADRetrievalIdea1_runtime'
code=Path('C:/Users/liuza/.codex/worktrees/text2cad-g5-cpu/CADRetrievalIdea1')
source=rt/'g5_complete_archive_bundle_20261003';control=rt/'g5_complete_archive_control_20261003'
target='/data/obs_data_liuza/CADRetrieval/runs/idea01_text2cad_g5_cpu_20261003'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','cvteam6']
source.mkdir(exist_ok=False);control.mkdir(exist_ok=False)

def save(name,obj):
    (control/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8',newline='\n')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def capture(name,argv,stdin=None):
    save(name+'_command.json',dict(argv=argv,pid=os.getpid()))
    r=subprocess.run(argv,input=stdin,capture_output=True)
    (control/(name+'_stdout.log')).write_bytes(r.stdout);(control/(name+'_stderr.log')).write_bytes(r.stderr)
    save(name+'_exit.json',dict(actual_exit_code=r.returncode))
    if r.returncode: raise RuntimeError(name+' exit '+str(r.returncode))
    return r.stdout

def checked_copytree(src,dst):
    raw=(src/'SHA256SUMS').read_bytes();assert b'\r' not in raw
    entries={n:h for h,n in (l.split('  ',1) for l in raw.decode().splitlines())}
    assert len(entries)==len(raw.decode().splitlines()),'duplicate manifest entries'
    paths=list(src.rglob('*'))
    assert all(p.is_file() and not p.is_symlink() and p.parent==src for p in paths),'formal/audit outputs must be flat with no directories/symlinks'
    assert {p.relative_to(src).as_posix() for p in paths}==set(entries)|{'SHA256SUMS'},'exact source manifest coverage'
    for n,h in entries.items(): assert digest(src/n)==h,n
    shutil.copytree(src,dst)
    assert (dst/'SHA256SUMS').read_bytes()==raw,'copied original manifest bytes differ'
    copied=list(dst.rglob('*'))
    assert all(p.is_file() and not p.is_symlink() and p.parent==dst for p in copied)
    assert {p.relative_to(dst).as_posix() for p in copied}==set(entries)|{'SHA256SUMS'}
    for n,h in entries.items(): assert digest(dst/n)==h,'copied payload differs '+n

def stream_archive(sender_argv,receiver_argv,se,so,re):
    # One ownership path: every spawned child is reaped before any final evidence write.
    sender=None;receiver=None
    save('pipeline_command.json',dict(sender_argv=sender_argv,receiver_argv=receiver_argv))
    try:
        sender=subprocess.Popen(sender_argv,stdout=subprocess.PIPE,stderr=se)
        save('sender_launch.json',dict(pid=sender.pid,argv=sender_argv))
        receiver=subprocess.Popen(receiver_argv,stdin=sender.stdout,stdout=so,stderr=re)
        sender.stdout.close()
        save('receiver_launch.json',dict(pid=receiver.pid,argv=receiver_argv))
        receiver.wait();sender.wait()
    finally:
        if sender is not None and sender.stdout is not None and not sender.stdout.closed:
            sender.stdout.close()
        for child in (receiver,sender):
            if child is not None:
                if child.poll() is None: child.terminate()
                child.wait()
        save('pipeline_exit.json',dict(sender_spawned=sender is not None,receiver_spawned=receiver is not None,
             sender_pid=None if sender is None else sender.pid,receiver_pid=None if receiver is None else receiver.pid,
             sender_exit=None if sender is None else sender.returncode,receiver_exit=None if receiver is None else receiver.returncode,
             sender_argv=sender_argv,receiver_argv=receiver_argv))
    if sender.returncode or receiver.returncode: raise RuntimeError('archive pipe failed')

try:
    audit=rt/'g5_independent_results_audit_20261003'
    assert json.loads((audit/'audit_result.json').read_text())['all_passed'] is True
    assert json.loads((audit/'actual_exit.json').read_text())['actual_child_exit_code']==0
    formal=repo/'experiments/runs/idea01_text2cad_g5_cpu_20261003'
    assert json.loads((formal/'actual_child_exit.json').read_text())['actual_child_exit_code']==0
    checked_copytree(formal,source/'formal');checked_copytree(audit,source/'independent_audit')
    proof=rt/'g5_fresh_review_20261003/review_e2d823_001'
    proof_manifest=json.loads((proof/'review_witness_manifest.json').read_text())
    assert {p.name for p in proof.iterdir()}==set(proof_manifest['witnesses'])|{'review_witness_manifest.json'}
    for n,pin in proof_manifest['witnesses'].items():
        assert (proof/n).is_file() and not (proof/n).is_symlink() and digest(proof/n)==pin['sha256'],n
    shutil.copytree(proof,source/'code_review')
    for n,pin in proof_manifest['witnesses'].items(): assert digest(source/'code_review'/n)==pin['sha256'],'copied review proof '+n
    for n in ['g5_cpu_targeted_tests_20261003_stdout.log','g5_cpu_targeted_tests_20261003_stderr.log',
              'g5_cpu_targeted_tests_20261003_exit.json','g5_cpu_input_linkage_preflight_20261003.json',
              'g5_root_bootstrap_toy_oracle_20261003.json','g5_independent_witness_toy_20261003.json',
              'g5_cpu_operator_launcher_20261003.json','g5_independent_operator_launcher_20261003.json',
              'archive_g5_complete_results_20261003.py']:
        shutil.copy2(rt/n,source/n)
    head=capture('code_head',['git','-C',str(code),'rev-parse','HEAD']).decode().strip()
    assert head=='e2d823aa284c800906ed37c0a2d0272e052de96b'
    assert not capture('code_status',['git','-C',str(code),'status','--porcelain','--untracked-files=all']).strip()
    code_binding={}
    for n in ['experiments/src/text2cad_g5_cpu.py','experiments/scripts/run_text2cad_g5_cpu.py',
              'experiments/configs/text2cad_g5_cpu_frozen_20261003.json','experiments/tests/test_text2cad_g5_cpu.py']:
        executed=(code/n).read_bytes();blob=subprocess.run(['git','-C',str(code),'show',head+':'+n],capture_output=True,check=True).stdout
        assert executed.replace(b'\r\n',b'\n')==blob,'approved canonical Git bytes differ '+n
        code_binding[n]=dict(executed_byte_sha256=hashlib.sha256(executed).hexdigest(),git_blob_sha256=hashlib.sha256(blob).hexdigest(),
             normalization_for_git_comparison='CRLF to LF only; executed original bytes preserved')
        dst=source/'approved_code'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(code/n,dst)
        assert dst.read_bytes()==executed,'copied executed source bytes differ '+n
    assert code_binding['experiments/src/text2cad_g5_cpu.py']['executed_byte_sha256']==json.loads((formal/'startup.json').read_text())['source_module_sha256']
    assert code_binding['experiments/configs/text2cad_g5_cpu_frozen_20261003.json']['executed_byte_sha256']==json.loads((formal/'startup.json').read_text())['config_sha256']
    (source/'approved_code_binding.json').write_text(json.dumps(dict(head=head,files=code_binding),indent=2)+'\n',encoding='utf-8')
    for n in ['experiments/plans/IDEA01_M1C_G5_CPU_EVALUATION_2026-10-03.md',
              'experiments/reviews/IDEA01_M1C_G4_MATRIX_CERTIFICATE_2026-10-03.json',
              'experiments/reviews/IDEA01_M1C_G5_CODEX_CODE_APPROVAL_2026-10-03.md',
              'experiments/submissions/idea01_text2cad_benchmark_20260928/CODEX_G5_CPU_LAUNCH_2026-10-03.md']:
        dst=source/'protocol'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/n,dst)
    config=json.loads((code/'experiments/configs/text2cad_g5_cpu_frozen_20261003.json').read_text())
    for n,p in config['inputs'].items():
        src=(repo if p['root']=='data' else rt)/p['path'];assert digest(src)==p['sha256'],n
        dst=source/'frozen_inputs'/n;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
        assert digest(dst)==p['sha256'],'copied frozen input differs '+n
    shutil.copy2(repo/'experiments/runs/idea01_text2cad_g4_full_20261002/SHA256SUMS',source/'G4_SOURCE_SHA256SUMS')
    assert digest(source/'G4_SOURCE_SHA256SUMS')=='97f9cbb60b0de449914ef1bdb2e388ddf0b2bcdae3686c73ceea511e764d3733'
    assert digest(source/'protocol/experiments/reviews/IDEA01_M1C_G4_MATRIX_CERTIFICATE_2026-10-03.json')=='afad9371cc846ad7e636c4ae4b76740ea1b74f787d002602bb677e9bb834bfc6'
    entries={p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file()}
    manifest=''.join(h+'  '+n+'\n' for n,h in sorted(entries.items())).encode();(source/'SHA256SUMS').write_bytes(manifest)
    manifest_hash=hashlib.sha256(manifest).hexdigest();nbytes=sum((source/n).stat().st_size for n in entries)
    save('source.json',dict(files=len(entries),bytes=nbytes,manifest_sha256=manifest_hash,target=target,zero_npu=True))
    dirs=sorted({(source/n).parent.relative_to(source).as_posix() for n in entries})
    commands=['set -eu','umask 077','test ! -e '+shlex.quote(target),'mkdir -- '+shlex.quote(target)]
    commands+=['mkdir -p -- '+shlex.quote(target+'/'+d) for d in dirs]
    commands+=['test -d '+shlex.quote(target+'/'+d) for d in dirs]
    capture('prepare',ssh+['bash -s'],('\n'.join(commands)+'\n').encode())
    sender_argv=['C:/WINDOWS/system32/tar.EXE','-czf','-','-C',str(source),'.']
    receiver_argv=ssh+['tar -xzf - --no-same-owner -C '+shlex.quote(target)]
    with (control/'sender_stderr.log').open('wb') as se,(control/'receiver_stdout.log').open('wb') as so,(control/'receiver_stderr.log').open('wb') as re:
        stream_archive(sender_argv,receiver_argv,se,so,re)
    capture('remote_sha',ssh+['cd '+shlex.quote(target)+' && LC_ALL=C sha256sum -c SHA256SUMS'])
    witness='''import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);expected=sys.argv[2]
raw=(r/'SHA256SUMS').read_bytes();assert hashlib.sha256(raw).hexdigest()==expected and b'\\r' not in raw
entries={n:h for h,n in (l.split('  ',1) for l in raw.decode().splitlines())}
assert {p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file()}==set(entries)|{'SHA256SUMS'}
total=0
for n,h in entries.items():
 p=r/n;assert not p.is_symlink() and r.resolve() in p.resolve().parents
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,n;total+=p.stat().st_size
print(json.dumps(dict(all_passed=True,verified_files=len(entries),payload_bytes=total,manifest_sha256=expected,unknown_extra=[])))
'''
    (control/'independent_readonly_witness.py').write_bytes(witness.encode())
    raw=capture('independent',ssh+['python3 -B - '+shlex.quote(target)+' '+manifest_hash],witness.encode())
    result=json.loads(raw);assert result['all_passed'] and result['verified_files']==len(entries) and result['payload_bytes']==nbytes
    save('completion.json',result);save('controller_exit.json',dict(actual_exit_code=0,pid=os.getpid(),ended_epoch=time.time()))
    print(raw.decode(),flush=True)
except BaseException as exc:
    save('controller_exit.json',dict(actual_exit_code=1,pid=os.getpid(),error=repr(exc),ended_epoch=time.time()))
    raise
