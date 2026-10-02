# Text2CAD G4：Git部署的零NPU Linux生命周期见证

2026-10-02。R3代码审查只批准本计划，不授权正式评分。CPU批准完整SHA：56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719（代码e8ef0cdd0349f255d8a299ae1b32fbf3fcdd1263不变）。GPU/模型/BERT/评分对数0；无下载或安装。只yunnan-share2既有环境spec10c18c17。

## 操作合同

1. 远端先记录现有HEAD/干净status/两卡npu-smi；脏树或未知运行冲突停止，不reset/clean。本地push后远端仅git fetch origin、干净detached指定CPU SHA并核对逐字一致。源代码、脚本、测试不在服务器编辑或SCP覆盖。
2. CPU证据目录为runtime/runs/m1c_g4_posix_cpu_20261002/，必须fresh。TMPDIR/HF_HOME/XDG_CACHE_HOME/输出均checkout外，解释器启动前PYTHONDONTWRITEBYTECODE=1。最多600秒墙钟，只测试创建的短CPU孩子及临时Git fixture；不创建正式G4run/ledger。
3. 以下诊断程序在本地写入本Git计划。它仅从获批checkout用runpy读已审test，包装观测方法打印清理前原始状态，不改变任何断言、信号或finally清理。使用标准库选22个必要test，不安装pytest、不导入模型/BERT/NPU。诊断命令本身不写服务器源码文件。
4. 保存完整stdout/stderr、真实exit、命令/环境名称/源文件SHA、HEAD/干净Git/npu-smi前后。正式评分仍不授权。test失败保留原始状态，区别活后代与zombie，按实际根因处理；不自动重跑、不换环境或放宽门。

## 固定只读诊断程序（经Git计划记录）

```python
import hashlib, json, os, platform, runpy, subprocess, sys, unittest
from pathlib import Path

repo = Path('/home/ma-user/work/CADRetrieval')
expected_sha = '56ac0f8e7e1c3680ab3e28bbe3bae9f1f95b0719'
head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)
if head != expected_sha or dirty:
    raise SystemExit('CPU witness refuses wrong SHA or dirty checkout')
test_path = repo / 'experiments/tests/test_text2cad_g4_launch.py'
ns = runpy.run_path(str(test_path), run_name='codex_g4_posix_cpu_witness')
tree_cls = ns['ProcessGroupTest']
original_observe = tree_cls._observe_tree_state
def observed(self, record, pid_file):
    state = original_observe(self, record, pid_file)
    print('\nCODEX_PRE_CLEANUP ' + json.dumps({'test': self.id(), 'observed': state, 'controller_record': record}, sort_keys=True), flush=True)
    return state
tree_cls._observe_tree_state = observed
print('CODEX_CPU_META ' + json.dumps({'head': head, 'platform': platform.platform(), 'python': sys.version, 'pid': os.getpid(), 'posix_groups': ns['g4']._PROCESS_GROUP_SUPPORTED, 'test_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest(), 'controller_sha256': hashlib.sha256((repo / 'experiments/src/idea01/text2cad_g4_launch.py').read_bytes()).hexdigest()}), flush=True)
loader = unittest.TestLoader()
suite = unittest.TestSuite()
for name in ['PostSpawnFaultTest', 'ProcessGroupTest', 'DescendantDeadlineTest', 'CommonDeadlineTest']:
    suite.addTests(loader.loadTestsFromTestCase(ns[name]))
for name in ['test_two_card_success_real_exit', 'test_one_card_fails_peer_not_misreported']:
    suite.addTest(ns['LaunchEngineTest'](name))
result = unittest.TextTestRunner(verbosity=2).run(suite)
torch_free = 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print('CODEX_CPU_RESULT ' + json.dumps({'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'torch_free': torch_free, 'successful': result.wasSuccessful()}), flush=True)
raise SystemExit(0 if result.wasSuccessful() and result.testsRun == 22 and not result.skipped and torch_free else 1)
```

调用使用/home/ma-user/work/envs/cadretrieval-idea01-torch271/bin/python -B -c，参数为上述固定Git计划中的程序，stdout/stderr写上述CPU证据目录。它不是正式controller/main/scorer命令，也不改变源码或评分实现。
