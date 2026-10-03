"""CPU only: byte preservation, unsafe inputs and actual pipeline exit codes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).parents[1] / 'scripts/run_g4_archive_recovery.py'
spec = importlib.util.spec_from_file_location('archive_recovery', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ArchiveRecoveryTests(unittest.TestCase):
    def test_authorization_binds_diagnostic_or_full_mode(self):
        approved = 'a' * 40
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / 'experiments').mkdir()
            active = repo / 'experiments/ACTIVE_TASK.md'
            active.write_text('`approved_sha`: `' + approved + '`\n`state`: `pilot_authorized`\n'
                              '`gpu_budget`: 0\n`max_scored_pairs`: 0\n`archive_mode`: `diagnostic`\n')
            def git_output(argv):
                if 'status' in argv:
                    return b''
                if 'show' in argv:
                    return MODULE.read_bytes()
                return (approved + '\n').encode()
            with patch.object(m, 'REPO', repo), patch.object(m, '__file__', str(repo / 'helper.py')), \
                    patch.object(m.subprocess, 'check_output', side_effect=git_output), \
                    patch.object(m, 'digest', return_value='b' * 64), \
                    patch.object(m.Path, 'read_bytes', return_value=MODULE.read_bytes()):
                self.assertEqual(m.verify_code_authorization(approved, 'diagnostic')['authorized_mode'], 'diagnostic')
                with self.assertRaises(ValueError):
                    m.verify_code_authorization(approved, 'full')

    def test_all_parent_directories_precede_receivers(self):
        entries = [('card0/a', '0' * 64, 1), ('card1/b', '0' * 64, 1),
                   ('outer_launch/deep/c', '0' * 64, 1), ('SHA256SUMS', '0' * 64, 1)]
        commands = m.destination_commands('/tmp/new archive', entries, True)
        self.assertEqual(commands[0], "mkdir -- '/tmp/new archive'")
        self.assertIn("'/tmp/new archive/card0'", commands[1])
        self.assertIn("'/tmp/new archive/card1'", commands[1])
        self.assertIn("'/tmp/new archive/outer_launch/deep'", commands[1])
        self.assertIn("test -d '/tmp/new archive/card0'", commands[2])
        self.assertEqual(m.destination_commands('/tmp/new', entries, False), ['mkdir -- /tmp/new'])

    def test_parent_preparation_rejects_escape(self):
        for name in ('../outside/file', 'a/../../file', 'x\\bad/file', '/x/file', '-I/file'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                m.destination_commands('/tmp/new', [(name, '0' * 64, 1)], True)

    def fixture(self, root, names=('card0/a.jsonl', 'card1/b.jsonl')):
        lines = []
        for i, name in enumerate(names):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(bytes([i + 1]) * (20 + i))
            lines.append(hashlib.sha256(p.read_bytes()).hexdigest() + '  ./' + name + '\n')
        raw = ''.join(lines).encode()
        (root / 'SHA256SUMS').write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_verified_exact_partition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sha = self.fixture(root)
            entries = m.read_manifest(root, sha, 2, 41)
            groups = m.partition(entries, 8)
            names = [e[0] for group in groups for e in group]
            self.assertEqual(len(groups), 2)
            self.assertCountEqual(names, ['card0/a.jsonl', 'card1/b.jsonl'])
            self.assertEqual(len(set(names)), 2)

    def test_wrong_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sha = self.fixture(root)
            (root / 'card0/a.jsonl').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                m.read_manifest(root, sha, 2, 41)

    def test_wrong_manifest_pin_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            with self.assertRaises(ValueError):
                m.read_manifest(root, '0' * 64, 2, 41)

    def test_unsafe_duplicate_and_self_paths_rejected(self):
        for name in ('../x', '/x', 'C:/x', '-I', 'a\\b', 'SHA256SUMS', './card0/a.jsonl'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.fixture(root)
                raw = (root / 'SHA256SUMS').read_bytes() + ('0' * 64 + '  ' + name + '\n').encode()
                (root / 'SHA256SUMS').write_bytes(raw)
                with self.assertRaises(ValueError):
                    m.read_manifest(root, hashlib.sha256(raw).hexdigest(), 3, 41)

    def test_occupied_control_directory_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                m.create_control(Path(d))

    def test_pipeline_keeps_both_actual_exit_codes(self):
        for producer_rc, consumer_rc in ((0, 0), (9, 0), (0, 7), (9, 7)):
            with self.subTest(p=producer_rc, c=consumer_rc), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                sink = root / 'received.bin'
                producer = [sys.executable, '-c', 'import sys; sys.stdout.buffer.write(b"abc"); sys.exit(' + str(producer_rc) + ')']
                consumer = [sys.executable, '-c', 'import sys,pathlib; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read()); sys.exit(' + str(consumer_rc) + ')', str(sink)]
                result = m.run_pipeline(producer, consumer, root, 's0')
                self.assertEqual(sink.read_bytes(), b'abc')
                self.assertEqual(result['sender_exit'], producer_rc)
                self.assertEqual(result['receiver_exit'], consumer_rc)
                self.assertEqual(result['ok'], producer_rc == consumer_rc == 0)
                self.assertEqual(json.loads((root / 's0_exit.json').read_text()), result)

    def test_receiver_spawn_failure_preserves_sender_pid_and_argv(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sender = [sys.executable, '-c', 'import time; time.sleep(20)']
            receiver = [str(root / 'nonexistent-receiver')]
            result = m.run_pipeline(sender, receiver, root, 'fault')
            self.assertFalse(result['ok'])
            self.assertIsNotNone(result['sender_exit'])
            self.assertIsNone(result['receiver_exit'])
            launch = json.loads((root / 'fault_launch.json').read_text())
            self.assertEqual(launch['sender_argv'], sender)
            self.assertEqual(launch['receiver_argv'], receiver)
            self.assertIsInstance(launch['sender_pid'], int)
            self.assertIsNone(launch['receiver_pid'])
            self.assertEqual(result['sender_pid'], launch['sender_pid'])


if __name__ == '__main__':
    unittest.main()
