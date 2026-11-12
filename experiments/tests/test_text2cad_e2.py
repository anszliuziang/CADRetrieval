"""Tests for the E2 empty-text background path (torch-free).

Covers: background-text admission (b0 empty string / b1 sentence, rejections),
trusted-origin allowlist, E2 ACTIVE_TASK authorization parsing, manifest-set
validation on toy fixtures (counts, cad-store equality, replay text pinning),
deadline enforcement, config loading, and dry-run refuse-overwrite.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01 import text2cad_e2_runner as e2  # noqa: E402
from idea01.p1_runner import (  # noqa: E402
    EXPECTED_ORIGIN_URL,
    EXPECTED_ORIGIN_URL_GITHUB_MIRROR,
    origin_allowed,
)
from idea01.text2cad_pair_index import (  # noqa: E402
    CAD_LENGTH_FILE,
    CAD_STORE_FILE,
    PAIR_MANIFEST_FILE,
    PairIndexError,
    SCHEMA_VERSION,
    TEXTS_FILE,
    load_background_texts,
    sha256_text,
)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jl(rows) -> bytes:
    return b"".join(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n" for r in rows)


def make_manifest(root: Path, texts: list[str], n_cads: int, lengths: bool = True) -> dict:
    """Write a toy manifest dir; returns fingerprints dict."""
    root.mkdir(parents=True, exist_ok=False)
    text_rows = [
        {"text_index": i, "text": t, "text_sha256": sha256_text(t), "n_chars": len(t)}
        for i, t in enumerate(texts)
    ]
    store_rows = [{"cad_index": i, "member": f"cad_seq/0000/{i:08d}/seq/{i:08d}.pth"} for i in range(n_cads)]
    length_rows = [
        {"cad_index": i, "n_real_tokens": 10 + i, "n_valid_x": 8, "n_valid_y": 2}
        for i in range(n_cads)
    ]
    files = {
        TEXTS_FILE: jl(text_rows),
        CAD_STORE_FILE: jl(store_rows),
        CAD_LENGTH_FILE: jl(length_rows) if lengths else b"",
    }
    for name, data in files.items():
        if data or lengths or name != CAD_LENGTH_FILE:
            (root / name).write_bytes(data)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {name: {"file": name, "sha256": sha(data)} for name, data in files.items()},
        "counts": {"n_unique_text": len(texts), "n_gallery": n_cads, "n_pairs": len(texts) * n_cads},
    }
    (root / PAIR_MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return {
        "texts_sha256": sha(files[TEXTS_FILE]),
        "cad_store_sha256": sha(files[CAD_STORE_FILE]),
        "cad_length_index_sha256": sha(files[CAD_LENGTH_FILE]),
        "pair_manifest_sha256": sha((root / PAIR_MANIFEST_FILE).read_bytes()),
    }


class TestBackgroundLoader(unittest.TestCase):
    def _dir(self, records):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        d = Path(td.name)
        rows = [
            {"text_index": i, "text": t, "text_sha256": sha256_text(t), "n_chars": len(t)}
            for i, t in records
        ]
        (d / TEXTS_FILE).write_bytes(jl(rows))
        return d

    def test_accepts_empty_string(self):
        d = self._dir([(0, "")])
        self.assertEqual(load_background_texts(d, ""), "")

    def test_accepts_sentence(self):
        d = self._dir([(0, "a CAD model.")])
        self.assertEqual(load_background_texts(d, "a CAD model."), "a CAD model.")

    def test_rejects_wrong_text(self):
        d = self._dir([(0, "hello")])
        with self.assertRaises(PairIndexError):
            load_background_texts(d, "")

    def test_rejects_two_records(self):
        d = self._dir([(0, ""), (1, "")])
        with self.assertRaises(PairIndexError):
            load_background_texts(d, "")

    def test_rejects_bad_index(self):
        d = self._dir([(3, "")])
        with self.assertRaises(PairIndexError):
            load_background_texts(d, "")


class TestOriginAllowlist(unittest.TestCase):
    def test_allowlist(self):
        self.assertTrue(origin_allowed(EXPECTED_ORIGIN_URL))
        self.assertTrue(origin_allowed(EXPECTED_ORIGIN_URL_GITHUB_MIRROR))
        self.assertTrue(origin_allowed("https://github.com/anszliuziang/CADRetrieval.git"))
        self.assertTrue(origin_allowed("HTTPS://CNB.COOL/lzacode/CADRetrieval/"))
        self.assertFalse(origin_allowed("https://example.com/attacker"))
        self.assertFalse(origin_allowed("https://cnb.cool/other/repo"))

    def test_explicit_expected_still_exact(self):
        # regression guard: explicit expected_origin keeps exact-match semantics
        text = "x"
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "remote", "add", "origin", EXPECTED_ORIGIN_URL_GITHUB_MIRROR], cwd=repo, check=True)
            from idea01.p1_runner import fetch_authorized_active_task

            _, meta = fetch_authorized_active_task(repo, expected_origin=EXPECTED_ORIGIN_URL)
            self.assertTrue(any("not authorized" in p for p in meta["problems"]))
            _, meta2 = fetch_authorized_active_task(repo, expected_origin=EXPECTED_ORIGIN_URL_GITHUB_MIRROR)
            # fetch will fail (no commits) but the ORIGIN check must pass
            self.assertFalse(any("origin remote" in p for p in meta2["problems"]))


AT_OK = """# 当前实验任务

- `state`: `pilot_authorized`
- `e2_approved_sha`: `{sha}`
- `e2_max_pairs`: 24285
- `e2_max_card_seconds`: 900
- `e2_deadline_utc`: 2026-11-12T18:00:00Z
"""


class TestE2AuthorizationParsing(unittest.TestCase):
    SHA = "a" * 40

    def test_ok(self):
        auth = e2.parse_e2_authorization(AT_OK.format(sha=self.SHA), self.SHA)
        self.assertEqual(auth["approved_sha"], self.SHA)
        self.assertEqual(auth["max_pairs"], 24285)
        self.assertEqual(auth["max_card_seconds"], 900.0)
        self.assertGreater(auth["deadline_epoch"], 0)

    def test_state_refused(self):
        bad = AT_OK.format(sha=self.SHA).replace("pilot_authorized", "implementation")
        with self.assertRaises(e2.E2Error):
            e2.parse_e2_authorization(bad, self.SHA)

    def test_sha_mismatch_refused(self):
        with self.assertRaises(e2.E2Error):
            e2.parse_e2_authorization(AT_OK.format(sha=self.SHA), "b" * 40)

    def test_missing_fields_refused(self):
        text = "- `state`: `pilot_authorized`\n"
        with self.assertRaises(e2.E2Error):
            e2.parse_e2_authorization(text, self.SHA)

    def test_bad_deadline_refused(self):
        bad = AT_OK.format(sha=self.SHA).replace("2026-11-12T18:00:00Z", "tomorrow")
        with self.assertRaises(e2.E2Error):
            e2.parse_e2_authorization(bad, self.SHA)

    def test_comma_separated_caps_accepted(self):
        text = AT_OK.format(sha=self.SHA).replace("24285", "24,285").replace("900", "900")
        auth = e2.parse_e2_authorization(text, self.SHA)
        self.assertEqual(auth["max_pairs"], 24285)

    def test_zero_cap_refused(self):
        bad = AT_OK.format(sha=self.SHA).replace("900", "0")
        with self.assertRaises(e2.E2Error):
            e2.parse_e2_authorization(bad, self.SHA)

    def test_deadline_enforced(self):
        auth = e2.parse_e2_authorization(AT_OK.format(sha=self.SHA), self.SHA)
        e2.check_deadline(auth, now=auth["deadline_epoch"] - 1)
        with self.assertRaises(e2.E2Error):
            e2.check_deadline(auth, now=auth["deadline_epoch"] + 1)


class TestE2ManifestValidation(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)

    def _cfg(self, original: Path, b0: Path, b1: Path, replay: Path, replay_texts):
        orig_store_sha = sha((original / CAD_STORE_FILE).read_bytes())
        # original cad_store rows MUST match make_manifest's member format so the
        # b0 store is byte-identical (the I1 fixture fix from the fresh review)
        cfg = {
            "schema": "text2cad-e2-background-expected-v1",
            "authorization_ref": "origin/glm-night-20261112",
            "backgrounds": {
                "b0": make_manifest(b0, [""], 3),
                "b1": make_manifest(b1, ["a CAD model."], 3),
                "smoke_b0": make_manifest(b1.parent / "smoke", [""], 1),
                "replay": make_manifest(replay, [t for _, t in replay_texts], 2),
            },
            "original_cad_store_sha256": orig_store_sha,
            "replay_texts": [
                {"original_text_index": i, "text": t, "text_sha256": sha256_text(t)}
                for i, t in replay_texts
            ],
            "cad_seq_zip_sha256": "0" * 64,
        }
        for which, n_t, n_c in (("b0", 1, 3), ("b1", 1, 3), ("smoke_b0", 1, 1), ("replay", 2, 2)):
            cfg["backgrounds"][which]["n_unique_text"] = n_t
            cfg["backgrounds"][which]["n_gallery"] = n_c
        return cfg

    def test_valid_set_passes(self):
        original = self.root / "orig"
        original.mkdir()
        (original / CAD_STORE_FILE).write_bytes(jl([{"cad_index": i, "member": f"m{i}"} for i in range(3)]))
        replay_texts = [(0, "text zero"), (1, "text one")]
        cfg = self._cfg(original, self.root / "b0", self.root / "b1", self.root / "replay", replay_texts)
        report = e2.validate_e2_manifests(
            cfg, {"b0": self.root / "b0", "b1": self.root / "b1", "smoke_b0": self.root / "smoke", "replay": self.root / "replay"}, original
        )
        self.assertEqual(report["backgrounds"]["b0"]["text"], "")
        self.assertEqual(report["backgrounds"]["replay"]["n_texts"], 2)

    def test_cad_store_mismatch_refused(self):
        original = self.root / "orig"
        original.mkdir()
        (original / CAD_STORE_FILE).write_bytes(jl([{"cad_index": i, "member": f"cad_seq/0000/{i:08d}/seq/{i:08d}.pth"} for i in range(3)]))
        replay_texts = [(0, "a"), (1, "b")]
        cfg = self._cfg(original, self.root / "b0", self.root / "b1", self.root / "replay", replay_texts)
        # tamper the ORIGINAL store after config pinning: its bytes must still
        # match the committed pin, otherwise refuse (candidate-drift gate)
        (original / CAD_STORE_FILE).write_bytes(jl([{"cad_index": i, "member": f"OTHER{i}"} for i in range(3)]))
        with self.assertRaises(e2.E2Error):
            e2.validate_e2_manifests(
                cfg, {"b0": self.root / "b0", "b1": self.root / "b1", "smoke_b0": self.root / "smoke", "replay": self.root / "replay"}, original
            )

    def test_replay_text_mismatch_refused(self):
        original = self.root / "orig"
        original.mkdir()
        (original / CAD_STORE_FILE).write_bytes(jl([{"cad_index": i, "member": f"m{i}"} for i in range(3)]))
        replay_texts = [(0, "pinned text"), (1, "pinned two")]
        cfg = self._cfg(original, self.root / "b0", self.root / "b1", self.root / "replay", replay_texts)
        # b0/b1/replay built with different replay texts than pinned
        cfg["backgrounds"]["replay"] = {
            **make_manifest(self.root / "replay2", ["changed"], 2),
            "n_unique_text": 2,
            "n_gallery": 2,
        }
        with self.assertRaises(e2.E2Error):
            e2.validate_e2_manifests(
                cfg, {"b0": self.root / "b0", "b1": self.root / "b1", "smoke_b0": self.root / "smoke", "replay": self.root / "replay2"}, original
            )

    def test_b0_needs_background_loader(self):
        # generic load_texts rejects the empty string; only the background
        # loader admits it — pin that contract
        b0 = self.root / "b0x"
        make_manifest(b0, [""], 3)
        from idea01.text2cad_pair_index import load_texts

        with self.assertRaises(PairIndexError):
            load_texts(b0)
        self.assertEqual(load_background_texts(b0, ""), "")


class TestDryRunRefusals(unittest.TestCase):
    def test_refuse_overwrite_first(self):
        # refusal guards precede validation (fresh review I1): empty fixtures are fine,
        # the existing dry_run.json alone triggers the refusal (out_dir outside the repo)
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            out = Path(td) / "dr"
            out.mkdir()
            (out / "dry_run.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(e2.E2Error) as ctx:
                e2.dry_run_e2(
                    cfg={"backgrounds": {}, "original_cad_store_sha256": "", "replay_texts": [], "cad_seq_zip_sha256": ""},
                    dirs={"b0": Path(td), "b1": Path(td), "replay": Path(td), "smoke_b0": Path(td)},
                    original_manifest_dir=Path(td),
                    bert_dir=Path(td),
                    cad_seq_zip=Path(td),
                    out_dir=out,
                    repo_root=repo,
                )
            self.assertIn("refusing to overwrite", str(ctx.exception))

    def test_refuse_outside_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            (repo / "experiments").mkdir(parents=True)
            out = repo / "experiments" / "runs" / "x"
            out.mkdir(parents=True)
            with self.assertRaises(e2.E2Error) as ctx:
                e2.dry_run_e2(
                    cfg={"backgrounds": {}, "original_cad_store_sha256": "", "replay_texts": [], "cad_seq_zip_sha256": ""},
                    dirs={"b0": Path(td), "b1": Path(td), "replay": Path(td), "smoke_b0": Path(td)},
                    original_manifest_dir=Path(td),
                    bert_dir=Path(td),
                    cad_seq_zip=Path(td),
                    out_dir=out,
                    repo_root=repo,
                )
            self.assertIn("inside the Git checkout", str(ctx.exception))


class TestConfigLoader(unittest.TestCase):
    def test_missing_config_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(e2.E2Error):
                e2.load_e2_config(Path(td))


if __name__ == "__main__":
    unittest.main()
