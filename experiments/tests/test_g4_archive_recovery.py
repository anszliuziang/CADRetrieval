"""CPU only: byte preservation, unsafe inputs and actual pipeline exit codes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / 'scripts/run_g4_archive_recovery.py'
spec = importlib.util.spec_from_file_location('archive_recovery', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ArchiveRecoveryTests(unittest.TestCase):
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
