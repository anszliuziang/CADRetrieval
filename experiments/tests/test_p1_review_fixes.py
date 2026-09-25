"""Tests for the Codex review 2026-09-25 fixes (R1-R4):

R1  resume/cumulative budget: three-invocation staged protocol 1 -> +9 -> +404
    must yield exactly 414 unique records within the whole-run caps; tampered
    or duplicate existing records refuse BEFORE weights; no --resume refuses.
R2  verifier: reproduces the review's tampered pair (token-sum -2 vs record
    sum -1) as a FAILURE, plus every other tampering path.
R3  authorization parsing/gating; SHA256SUMS verification.
R4  budget clamping, out-of-checkout guard, unique invocation ids.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01.p1_runner import (  # noqa: E402
    HARD_CAPS,
    BudgetLedger,
    active_task_field,
    check_authorization,
    effective_budgets,
    ensure_outside_checkout,
    fetch_authorized_active_task,
    new_invocation_id,
    normalize_remote_url,
    parse_pair_ids,
    plan_invocation,
    verify_existing_records,
    verify_sha256sums,
)
from idea01.p1_verify import EOS_ID, verify_pair_record, verify_run  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

BINDING = {
    "selection_sha256": "s" * 64,
    "git_sha": "g" * 40,
    "model_dir": "/models/x",
    "data_dir": "/data/y",
    "model_revision": "mrev",
    "data_revision": "drev",
    "dtype": "bfloat16",
    "env_spec_canonical_json_sha256": "c" * 64,
    "env_spec_file_bytes_sha256": "f" * 64,
}
CAPS = {"max_forwards": 416, "max_wallclock_s": 2700.0, "max_card_hours": 1.0}


def mk_record(pair_id, status="ok", **over):
    rec = {
        **BINDING,
        "pair_id": pair_id,
        "status": status,
        "all_finite": True,
        "sum_logprob": -10.0,
    }
    rec.update(over)
    return rec


class TestR1ExistingRecords(unittest.TestCase):
    def test_clean_records_pass(self):
        recs = [mk_record(f"p{i:03d}") for i in range(3)]
        self.assertEqual(verify_existing_records(recs, BINDING), [])

    def test_duplicate_id_refused(self):
        recs = [mk_record("p000"), mk_record("p000")]
        self.assertTrue(any("duplicate" in p for p in verify_existing_records(recs, BINDING)))

    def test_failed_status_refused(self):
        recs = [mk_record("p000", status="error", all_finite=False)]
        self.assertTrue(any("not ok/finite" in p for p in verify_existing_records(recs, BINDING)))

    def test_mixed_version_refused(self):
        for field, bad in (
            ("selection_sha256", "x" * 64),
            ("git_sha", "x" * 40),
            ("dtype", "float16"),
            ("model_dir", "/elsewhere"),
            ("data_revision", "other"),
            ("env_spec_canonical_json_sha256", "0" * 64),
        ):
            recs = [mk_record("p000", **{field: bad})]
            probs = verify_existing_records(recs, BINDING)
            self.assertTrue(any(field in p for p in probs), field)


class TestR1StagedProtocol(unittest.TestCase):
    """Simulate the full 1 -> +9 -> +404 staged protocol with pure functions."""

    def setUp(self):
        self.pairs = [{"pair_id": f"p{i:03d}"} for i in range(414)]
        self.td = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.td.name) / "budget_ledger.json"

    def tearDown(self):
        self.td.cleanup()

    def _invocation(self, existing, spec, resume, wall, fwd_per_pair=1):
        """One simulated invocation; returns (to_score, skipped) or raises."""
        probs = verify_existing_records(existing, BINDING)
        self.assertEqual(probs, [])
        requested = parse_pair_ids(spec, len(self.pairs), 416)
        to_score, skipped = plan_invocation(existing, requested, self.pairs, resume)
        ledger = BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], CAPS)
        self.assertEqual(ledger.check_can_start(len(to_score)), [])
        ledger.begin("inv")
        ledger.finalize("inv", wall, len(to_score) * fwd_per_pair, 0)
        return to_score, skipped

    def test_one_then_nine_then_404(self):
        records = []
        s1, sk1 = self._invocation(records, "0", False, 30.0)
        self.assertEqual((s1, sk1), ([0], []))
        records += [mk_record("p000")]

        s2, sk2 = self._invocation(records, "0-9", True, 60.0)
        self.assertEqual(s2, list(range(1, 10)))
        self.assertEqual(sk2, ["p000"])
        records += [mk_record(f"p{i:03d}") for i in range(1, 10)]

        s3, sk3 = self._invocation(records, "all", True, 600.0)
        self.assertEqual(s3, list(range(10, 414)))
        self.assertEqual(len(s3), 404)
        self.assertEqual(len(sk3), 10)
        records += [mk_record(f"p{i:03d}") for i in range(10, 414)]

        # exactly 414 unique raw records; cumulative forwards 414 <= 416
        ids = [r["pair_id"] for r in records]
        self.assertEqual(len(ids), len(set(ids)), 414)
        ledger = BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], CAPS)
        self.assertEqual(ledger.cumulative(), (414, 690.0))
        # a further invocation beyond the cap is refused; exactly up to it is allowed
        self.assertTrue(any("forwards budget" in p for p in ledger.check_can_start(3)))
        self.assertEqual(ledger.check_can_start(2), [])  # 414+2 == 416 ok

    def test_no_resume_with_existing_refused(self):
        with self.assertRaises(ValueError):
            plan_invocation([mk_record("p000")], [0, 1], self.pairs, resume=False)

    def test_tampered_existing_refused_before_weights(self):
        records = [mk_record("p000", selection_sha256="t" * 64)]
        self.assertTrue(verify_existing_records(records, BINDING))

    def test_unfinalized_ledger_refuses(self):
        ledger = BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], CAPS)
        ledger.begin("crashy")  # never finalized: simulated crash
        with self.assertRaises(ValueError):
            BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], CAPS)

    def test_ledger_binding_mismatch(self):
        ledger = BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], CAPS)
        ledger.begin("a")
        ledger.finalize("a", 1.0, 1, 0)
        with self.assertRaises(ValueError):
            BudgetLedger.load(self.ledger_path, "z" * 64, CAPS)
        with self.assertRaises(ValueError):
            BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], {**CAPS, "max_forwards": 100})

    def test_wallclock_cap_binds(self):
        ledger = BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], CAPS)
        ledger.begin("a")
        ledger.finalize("a", 2700.0, 10, 0)
        self.assertTrue(any("wallclock" in p for p in ledger.check_can_start(1)))
        self.assertEqual(ledger.budget_exceeded(0.1), "max_wallclock_s")

    def test_cardhour_cap_binds(self):
        caps = {"max_forwards": 416, "max_wallclock_s": 1e9, "max_card_hours": 0.001}
        ledger = BudgetLedger.load(self.ledger_path, BINDING["selection_sha256"], caps)
        ledger.begin("a")
        ledger.finalize("a", 3.59, 1, 0)  # 0.000997 card-hours
        self.assertIsNone(ledger.budget_exceeded(0.0))
        self.assertEqual(ledger.budget_exceeded(0.02), "max_card_hours")


def mk_pair_evidence(sum_lp=-10.0, n_prefix=3, n_target=4):
    """Consistent (record, tokens) pair; tamper from here in tests."""
    ids = [100, 101, 102] + [200, 201, 202, EOS_ID]
    per_tok = sum_lp / n_target
    toks = [
        {"position": n_prefix + k, "pred_row": n_prefix + k - 1, "token_id": ids[n_prefix + k], "logprob": per_tok}
        for k in range(n_target)
    ]
    rec = {
        "pair_id": "p000",
        "status": "ok",
        "all_finite": True,
        "input_ids": ids,
        "n_prefix_tokens": n_prefix,
        "n_input_tokens": len(ids),
        "n_target_tokens": n_target,
        "sum_logprob": sum_lp,
        "mean_logprob": sum_lp / n_target,
        "eos_logprob": per_tok,
    }
    return rec, toks


class TestR2Verifier(unittest.TestCase):
    def test_good_pair_passes(self):
        rec, toks = mk_pair_evidence()
        self.assertEqual(verify_pair_record(rec, toks), [])

    def test_review_repro_sum_tampered_fails(self):
        """The review's exact case: token-sum -2 vs record sum -1 must FAIL."""
        rec, toks = mk_pair_evidence(sum_lp=-2.0)
        rec["sum_logprob"] = -1.0
        rec["mean_logprob"] = -0.25
        errs = verify_pair_record(rec, toks)
        self.assertTrue(any("token-sum" in e for e in errs), errs)
        report = verify_run([rec], {"p000": toks}, ["p000"])
        self.assertFalse(report["ok"])
        self.assertGreater(report["n_failures"], 0)

    def test_token_id_vs_input_ids(self):
        rec, toks = mk_pair_evidence()
        toks[1]["token_id"] += 1
        self.assertTrue(any("token_id" in e for e in verify_pair_record(rec, toks)))

    def test_position_not_contiguous(self):
        rec, toks = mk_pair_evidence()
        toks[2]["position"] += 1
        self.assertTrue(any("contiguous" in e for e in verify_pair_record(rec, toks)))

    def test_pred_row_shift(self):
        rec, toks = mk_pair_evidence()
        toks[0]["pred_row"] = toks[0]["position"]
        self.assertTrue(any("pred_row" in e for e in verify_pair_record(rec, toks)))

    def test_missing_and_empty_token_file(self):
        rec, _ = mk_pair_evidence()
        self.assertTrue(any("missing token" in e for e in verify_pair_record(rec, None)))
        self.assertTrue(any("empty token" in e for e in verify_pair_record(rec, [])))

    def test_last_token_not_eos(self):
        rec, toks = mk_pair_evidence()
        toks[-1]["token_id"] = 999
        self.assertTrue(verify_pair_record(rec, toks))

    def test_input_ids_checks(self):
        rec, toks = mk_pair_evidence()
        rec["input_ids"][-1] = 999
        self.assertTrue(any("EOS" in e for e in verify_pair_record(rec, toks)))
        rec, toks = mk_pair_evidence()
        rec["n_input_tokens"] += 1
        self.assertTrue(any("n_input_tokens" in e for e in verify_pair_record(rec, toks)))
        rec, toks = mk_pair_evidence()
        rec["n_target_tokens"] += 1
        self.assertTrue(verify_pair_record(rec, toks))

    def test_mean_and_eos_logprob(self):
        rec, toks = mk_pair_evidence()
        rec["mean_logprob"] = 0.0
        self.assertTrue(any("mean_logprob" in e for e in verify_pair_record(rec, toks)))
        rec, toks = mk_pair_evidence()
        rec["eos_logprob"] = 0.0
        self.assertTrue(any("eos_logprob" in e for e in verify_pair_record(rec, toks)))

    def test_nonfinite_and_bad_status(self):
        rec, toks = mk_pair_evidence()
        rec["all_finite"] = False
        self.assertTrue(any("all_finite" in e for e in verify_pair_record(rec, toks)))
        rec, toks = mk_pair_evidence()
        rec["status"] = "error"
        self.assertTrue(verify_pair_record(rec, toks))

    def test_run_level_duplicate_missing_extra(self):
        rec, toks = mk_pair_evidence()
        self.assertTrue(verify_run([rec], {"p000": toks}, ["p000"])["ok"])
        self.assertFalse(verify_run([rec, rec], {"p000": toks}, ["p000"])["ok"])  # dup
        self.assertFalse(verify_run([], {}, ["p000"])["ok"])  # missing
        self.assertFalse(verify_run([rec], {"p000": toks}, ["p999"])["ok"])  # extra+missing


AT_TMPL = """# 当前实验任务

- `state`: `{state}`
- `task_id`: `idea01_p1_conditional_sensitivity_20260925`
- `approved_sha`: {sha_line}
- `approved_selection_sha256`: {sel_line}
"""


class TestR3Authorization(unittest.TestCase):
    def test_field_parsing(self):
        text = AT_TMPL.format(state="pilot_authorized", sha_line="`" + "a" * 40 + "`（已审核）", sel_line="`" + "b" * 64 + "`")
        self.assertEqual(active_task_field(text, "state"), "pilot_authorized")
        self.assertEqual(active_task_field(text, "approved_sha"), "a" * 40)
        self.assertEqual(active_task_field(text, "approved_selection_sha256"), "b" * 64)
        self.assertIsNone(active_task_field(text, "nonexistent"))

    def test_authorized_passes(self):
        text = AT_TMPL.format(state="pilot_authorized", sha_line="`" + "a" * 40 + "`", sel_line="`" + "b" * 64 + "`")
        self.assertEqual(
            check_authorization(text, "a" * 40, "b" * 64, "idea01_p1_conditional_sensitivity_20260925"),
            [],
        )

    def test_not_authorized_refused(self):
        text = AT_TMPL.format(state="review_required", sha_line="无", sel_line="无")
        probs = check_authorization(text, "a" * 40, "b" * 64, "idea01_p1_conditional_sensitivity_20260925")
        self.assertTrue(any("pilot_authorized" in p for p in probs))

    def test_wrong_sha_or_missing_pin_refused(self):
        text = AT_TMPL.format(state="pilot_authorized", sha_line="`" + "c" * 40 + "`", sel_line="`" + "b" * 64 + "`")
        probs = check_authorization(text, "a" * 40, "b" * 64, "idea01_p1_conditional_sensitivity_20260925")
        self.assertTrue(any("approved_sha" in p for p in probs))
        text = AT_TMPL.format(state="pilot_authorized", sha_line="`" + "a" * 40 + "`", sel_line="无")
        probs = check_authorization(text, "a" * 40, "b" * 64, "idea01_p1_conditional_sensitivity_20260925")
        self.assertTrue(any("approved_selection_sha256" in p for p in probs))
        text = AT_TMPL.format(state="pilot_authorized", sha_line="`" + "a" * 40 + "`", sel_line="`" + "d" * 64 + "`")
        probs = check_authorization(text, "a" * 40, "b" * 64, "idea01_p1_conditional_sensitivity_20260925")
        self.assertTrue(any("selection file sha256" in p for p in probs))

    def test_current_active_task_refuses(self):
        """The real ACTIVE_TASK (review_required, no approved_sha) must refuse."""
        text = (REPO / "experiments" / "ACTIVE_TASK.md").read_text(encoding="utf-8")
        probs = check_authorization(text, "a" * 40, "b" * 64, "idea01_p1_conditional_sensitivity_20260925")
        self.assertTrue(any("pilot_authorized" in p for p in probs))


class TestR3OriginAuthorization(unittest.TestCase):
    """Authorization is read from origin/master after fetch (review addendum:
    a commit cannot contain its own approved_sha, so the detached HEAD's
    working-tree ACTIVE_TASK can never prove authorization)."""

    def _git(self, cwd, *args):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f"git {' '.join(args)}: {r.stderr}")
        return r.stdout.strip()

    def _make_remote(self):
        """Two-commit 'Codex' repo: commit1 = code, commit2 = ACTIVE_TASK
        authorizing commit1's sha. Returns (bare, work, code_sha)."""
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        src = Path(td.name) / "src"
        src.mkdir()
        self._git(src, "init", "-q")
        self._git(src, "config", "user.email", "t@example.com")
        self._git(src, "config", "user.name", "t")
        (src / "code.txt").write_text("code v1", encoding="utf-8")
        self._git(src, "add", ".")
        self._git(src, "commit", "-qm", "code")
        code_sha = self._git(src, "rev-parse", "HEAD")
        (src / "experiments").mkdir()
        (src / "experiments" / "ACTIVE_TASK.md").write_text(
            AT_TMPL.format(
                state="pilot_authorized",
                sha_line="`" + code_sha + "`",
                sel_line="`" + "b" * 64 + "`",
            ),
            encoding="utf-8",
        )
        self._git(src, "add", ".")
        self._git(src, "commit", "-qm", "authorize")
        bare = Path(td.name) / "remote.git"
        self._git(src, "clone", "-q", "--bare", str(src), str(bare))
        work = Path(td.name) / "work"
        self._git(Path(td.name), "clone", "-q", str(bare), str(work))
        # consumer runs the CODE commit detached, not the authorization commit
        self._git(work, "checkout", "-q", "--detach", code_sha)
        return bare, work, code_sha

    def test_fetch_and_authorize_from_origin_master(self):
        bare, work, code_sha = self._make_remote()
        text, meta = fetch_authorized_active_task(work, expected_origin=str(bare))
        self.assertEqual(meta["problems"], [])
        self.assertIsNotNone(text)
        self.assertIn("origin_ref_sha", meta)
        self.assertNotEqual(meta["origin_ref_sha"], code_sha)  # authorization commit != code commit
        probs = check_authorization(text, code_sha, "b" * 64, "idea01_p1_conditional_sensitivity_20260925")
        self.assertEqual(probs, [])

    def test_detached_head_alone_would_refuse(self):
        # the code commit's OWN working tree has no ACTIVE_TASK at all; even if
        # it had one, it could not name its own sha (self-reference)
        _bare, work, code_sha = self._make_remote()
        wt = work / "experiments" / "ACTIVE_TASK.md"
        self.assertFalse(wt.exists(), "code commit predates the authorization commit")

    def test_wrong_origin_refused(self):
        bare, work, _ = self._make_remote()
        text, meta = fetch_authorized_active_task(work, expected_origin="https://example.com/attacker")
        self.assertIsNone(text)
        self.assertTrue(any("expected" in p for p in meta["problems"]))

    def test_normalize_url(self):
        self.assertEqual(normalize_remote_url("https://h/a/b.git"), normalize_remote_url("https://h/a/b/"))
        self.assertEqual(normalize_remote_url("HTTPS://H/A/B"), "https://h/a/b")


class TestR3Sha256Sums(unittest.TestCase):
    def _dir(self, files: dict[str, bytes], sums: str | None):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        for rel, data in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        if sums is not None:
            (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
        self.addCleanup(td.cleanup)
        return root

    def test_ok(self):
        import hashlib

        h1 = hashlib.sha256(b"alpha").hexdigest()
        h2 = hashlib.sha256(b"beta").hexdigest()
        root = self._dir({"a.bin": b"alpha", "sub/b.bin": b"beta"}, f"{h1}  a.bin\n{h2}  sub/b.bin\n")
        n_ok, fails = verify_sha256sums(root)
        self.assertEqual((n_ok, fails), (2, []))

    def test_tampered_file_detected(self):
        import hashlib

        h1 = hashlib.sha256(b"alpha").hexdigest()
        root = self._dir({"a.bin": b"ALPHA"}, f"{h1}  a.bin\n")
        n_ok, fails = verify_sha256sums(root)
        self.assertEqual(n_ok, 0)
        self.assertTrue(any("mismatch" in f for f in fails))

    def test_missing_file_and_missing_sums(self):
        import hashlib

        h1 = hashlib.sha256(b"alpha").hexdigest()
        root = self._dir({}, f"{h1}  gone.bin\n")
        _, fails = verify_sha256sums(root)
        self.assertTrue(any("missing file" in f for f in fails))
        root2 = self._dir({"a.bin": b"alpha"}, None)
        _, fails2 = verify_sha256sums(root2)
        self.assertTrue(any("missing" in f and "SHA256SUMS" in f for f in fails2))


class TestR4Guards(unittest.TestCase):
    def test_budget_clamp(self):
        b = effective_budgets({"max_forwards": 9999, "max_wallclock_s": 99999.0, "npu_card_hours": 99.0})
        self.assertEqual(b, HARD_CAPS)
        b2 = effective_budgets({"max_forwards": 100, "max_wallclock_s": 60.0, "npu_card_hours": 0.1})
        self.assertEqual(b2, {"max_forwards": 100, "max_wallclock_s": 60.0, "max_card_hours": 0.1})

    def test_outside_checkout(self):
        with self.assertRaises(ValueError):
            ensure_outside_checkout(REPO / "experiments" / "runs" / "x", REPO)
        with self.assertRaises(ValueError):
            ensure_outside_checkout(REPO, REPO)
        with tempfile.TemporaryDirectory() as td:
            ensure_outside_checkout(Path(td), REPO)  # outside: no raise

    def test_invocation_ids_unique(self):
        a, b = new_invocation_id(), new_invocation_id()
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
