import ast,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
src=Path('D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/archive_g5_complete_results_20261003.py')
tree=ast.parse(src.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('digest','checked_copytree','stream_archive')]
ns=dict(hashlib=hashlib,json=json,shutil=shutil,subprocess=subprocess,Path=Path)
exec(compile(ast.Module(body=functions,type_ignores=[]),str(src),'exec'),ns)
checks=[]
with tempfile.TemporaryDirectory(prefix='g5_archive_toy_',dir='D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/tmp') as tmp:
 root=Path(tmp);s=root/'src';s.mkdir();(s/'p').write_bytes(b'safe payload')
 h=hashlib.sha256((s/'p').read_bytes()).hexdigest();(s/'SHA256SUMS').write_text(h+'  p\n',encoding='utf-8',newline='\n')
 ns['checked_copytree'](s,root/'valid');checks.append('valid flat copy exact SHA')
 (s/'extra').mkdir();(s/'extra/p').write_bytes(b'outside manifest')
 try:ns['checked_copytree'](s,root/'bad_nested')
 except AssertionError:checks.append('recursive unknown file rejected before copy')
 else:raise AssertionError('nested extra accepted')
 shutil.rmtree(s/'extra')
 copytree=shutil.copytree
 def corruptcopy(a,b,*args,**kwargs):
  r=copytree(a,b,*args,**kwargs);(Path(b)/'p').write_bytes(b'corrupted copied payload');return r
 shutil.copytree=corruptcopy
 try:ns['checked_copytree'](s,root/'corrupted')
 except AssertionError:checks.append('post-copy bytes reject corruption')
 else:raise AssertionError('corrupted copy accepted')
 finally:shutil.copytree=copytree
 records={};fault=None
 def save(n,obj):
  if n==fault:raise OSError('toy evidence failure '+n)
  records[n]=obj
 ns['save']=save
 producer=[sys.executable,'-B','-c','import sys,time;sys.stdout.buffer.write(b"abc");sys.stdout.flush();time.sleep(30)']
 quick=[sys.executable,'-B','-c','import sys;sys.stdout.buffer.write(b"abc")']
 receiver=[sys.executable,'-B','-c','import sys;assert sys.stdin.buffer.read()==b"abc"']
 for label,send,receive,injected in [('receiver_missing',producer,[str(root/'no_such.exe')],None),('sender_record_fault',producer,receiver,'sender_launch.json'),('receiver_record_fault',producer,receiver,'receiver_launch.json'),('normal',quick,receiver,None),('nonzero_receiver',quick,[sys.executable,'-B','-c','import sys;sys.stdin.buffer.read();sys.exit(2)'],None)]:
  records.clear();fault=injected
  with (root/(label+'se')).open('wb') as se,(root/(label+'so')).open('wb') as so,(root/(label+'re')).open('wb') as re:
   failure=None
   try:ns['stream_archive'](send,receive,se,so,re)
   except Exception as exc:failure=type(exc).__name__
  r=records['pipeline_exit.json'];assert r['sender_spawned'] and r['sender_exit'] is not None
  if label=='normal':assert failure is None and r['sender_exit']==r['receiver_exit']==0
  else:assert failure is not None
  if label=='receiver_missing':assert not r['receiver_spawned'] and r['receiver_exit'] is None
  if label=='nonzero_receiver':assert r['receiver_exit']==2
  for pid in (r['sender_pid'],r['receiver_pid']):
   if pid is not None:
    v=subprocess.run(['tasklist','/FI','PID eq '+str(pid),'/FO','CSV','/NH'],capture_output=True).stdout.decode('utf-8',errors='replace')
    assert '"'+str(pid)+'"' not in v,'own child still live'
  checks.append(label+': actual exit captured, owned children gone before test cleanup')
result=dict(all_passed=True,checks=len(checks),cases=checks,helper_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),zero_npu=True)
Path('D:/projects/IdeaSpark/CADRetrievalIdea1_runtime/g5_archive_boundary_tests_20261003.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
