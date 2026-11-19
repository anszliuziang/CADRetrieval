import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idea01 import text2cad_e1_selection as select


def row(uid, project, program, vector):
    return {
        "uid": uid, "project": project, "program_sha256": program,
        "cad_vec_sha256": vector,
    }


def test_fusion_families_join_project_and_reencoded_duplicates_transitively():
    rows = [
        row("j2_A_1", "j2", "p1", "v1"),
        row("j1_A_1", "j1", "p1", "v2"),
        row("j1_A_2", "j1", "p2", "v3"),
        row("j3_A_1", "j3", "p3", "v3"),
        row("j4_A_1", "j4", "p4", "v4"),
    ]
    groups = select.candidate_groups("fusion360", rows)
    assert sorted(len(group) for group in groups.values()) == [1, 4]
    members = sorted(r["uid"] for r in rows[:-1])
    expected = hashlib.sha256(("fusion360\n" + "\n".join(members)).encode()).hexdigest()
    assert sorted(groups[expected]) == members


def test_domain_assignment_orders_groups_and_never_draws_from_holdout():
    rows = [row(f"{i:03d}", "ignored_on_source", f"p{i}", f"v{i}") for i in range(40)]
    mapping = select.candidate_groups("deepcad", rows)
    given = select.group_splits("deepcad", mapping)
    reverse = select.group_splits("deepcad", select.candidate_groups("deepcad", rows[::-1]))
    assert given == reverse
    available = {r["uid"] for r in given["eligible"]}
    held_out = {uid for r in given["reserve"] for uid in r["member_uids"]}
    assert available.isdisjoint(held_out)
    assert len(available | held_out) == len(rows)


def test_select_takes_same_registered_gallery_size():
    assert select.choose_size(777, 569) == 512
    assert select.choose_size(777, 290) == 256
    with pytest.raises(ValueError, match="256"):
        select.choose_size(777, 255)
