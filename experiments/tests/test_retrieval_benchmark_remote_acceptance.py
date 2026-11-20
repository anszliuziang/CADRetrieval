"""Regression checks for remote-only acceptance using real evaluator receipts."""

from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_retrieval_benchmark_remote_acceptance as remote
from prepare_retrieval_benchmark_fill import load_json, sha256
from test_retrieval_benchmark_first_certificate import completed_run  # noqa: F401


@pytest.fixture
def remote_run(completed_run, monkeypatch):
    run, _, pulls = completed_run
    for pull in pulls.values():
        shutil.copytree(pull / "raw/full", run / "full", dirs_exist_ok=True)
    shutil.copytree(pulls["l2"] / "raw/control/score", run / "control/score")
    output = run / "remote_acceptance_test"
    shutil.copytree(run / "frozen", output / "frozen")
    approval = load_json(run / "full/run_authorization.json")
    monkeypatch.setattr(remote, "load_pinned_approval", lambda: approval)
    monkeypatch.setattr(remote, "GALLERY_SIZE", 3)
    monkeypatch.setattr(remote, "CONDITION_COUNTS", {"ct_deepcad": (2, 3), "l2": (3, 3)})
    monkeypatch.setattr(remote, "PLANNED_PAIRS", 15)
    return run, output


def test_remote_certificate_binds_settled_raw_rows_and_both_rank_witnesses(remote_run):
    run, out = remote_run
    approval = remote.preflight(run, out)
    early = remote.evaluate(run, out, "ct_deepcad", approval)
    result = remote.accept(run, out, approval)
    assert result["passed"] and result["pairs_charged"] == 15
    assert result["ledger_sha256"] == sha256(run / "full/ledger.json")
    assert result["conditions"]["ct_deepcad"]["rank_audit_sha256"] == early["rank_audit_sha256"]
    assert set(result["conditions"]["l2"]["methods"]) == {"sum", "s48"}


def test_remote_certificate_rehashes_an_already_audited_ct_matrix(remote_run):
    run, out = remote_run
    approval = remote.preflight(run, out)
    remote.evaluate(run, out, "ct_deepcad", approval)
    row = run / "full/scores/ct_deepcad/t00000.npy"
    row.write_bytes(row.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="raw ct_deepcad score receipt"):
        remote.accept(run, out, approval)


def test_remote_certificate_refuses_an_unsuccessful_controller_before_ranking(remote_run):
    run, out = remote_run
    approval = remote.preflight(run, out)
    (run / "control/score/controller.actual_exit").write_text("2\n", encoding="ascii")
    with pytest.raises(ValueError, match="controller did not exit successfully"):
        remote.accept(run, out, approval)
    assert not (out / "evaluation").exists()
