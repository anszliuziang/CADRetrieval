"""Environment-spec hashing: two documented algorithms, kept distinct.

The Idea 01 env spec (`experiments/configs/idea01_yunnan_share2_env.json`) has
two hashes on record, from the same content:

- canonical JSON SHA256: parse the file, re-serialize with
  `json.dumps(obj, sort_keys=True, separators=(",", ":"))` (ensure_ascii
  default True), encode utf-8, sha256 hexdigest. Insensitive to key order and
  line endings. Preparation-phase record prefix: `10c18c17`.
- file-bytes SHA256: sha256 hexdigest of the raw file bytes as checked out by
  Git (LF endings). Runtime record prefix: `b6cdd884`. Note: on Windows
  worktrees with autocrlf the working-copy bytes differ (CRLF); use Git blob
  bytes or an LF checkout for this algorithm.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def canonical_json_sha256(path: str | Path) -> str:
    """SHA256 of the canonical JSON serialization (sort_keys, tight separators)."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def file_bytes_sha256(path: str | Path) -> str:
    """SHA256 of the file bytes on disk as-is."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha256(repo_root: str | Path, repo_rel_path: str) -> str:
    """SHA256 of the Git blob bytes (LF endings, as a clean Linux checkout sees them)."""
    blob = subprocess.run(
        ["git", "show", f"HEAD:{repo_rel_path}"], cwd=repo_root, capture_output=True, check=True
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def env_spec_hashes(path: str | Path, repo_root: str | Path | None = None, repo_rel_path: str | None = None) -> dict:
    """Both documented hashes for the env spec file.

    If repo_root/repo_rel_path are given, the file-bytes algorithm is computed
    on the Git blob bytes (LF), avoiding Windows CRLF working-copy artifacts;
    otherwise raw file bytes are used (correct on LF checkouts).
    """
    out = {
        "canonical_json_sha256": canonical_json_sha256(path),
        "canonical_json_algorithm": "sha256(utf8(json.dumps(obj, sort_keys=True, separators=(',',':'))))",
    }
    if repo_root is not None and repo_rel_path is not None:
        out["file_bytes_sha256"] = git_blob_sha256(repo_root, repo_rel_path)
        out["file_bytes_algorithm"] = "sha256(git blob bytes, LF endings)"
    else:
        out["file_bytes_sha256"] = file_bytes_sha256(path)
        out["file_bytes_algorithm"] = "sha256(file bytes as checked out)"
    return out
