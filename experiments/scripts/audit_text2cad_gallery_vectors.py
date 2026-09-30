"""Zero-forward structural coverage audit for the pinned Text2CAD test gallery.

This is an audit process, separate from the scorer. It may read split UIDs to
locate CAD files, but emits no query-to-candidate relevance information.
"""

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from idea01.text2cad_vector_scorer import ScorerInputError, validate_cad_vec


def quantiles(values: list[int]) -> dict[str, int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        str(percent): ordered[round((len(ordered) - 1) * percent / 100)]
        for percent in (0, 10, 25, 50, 75, 90, 95, 99, 100)
    }


def audit(split_path: Path, cad_zip_path: Path) -> dict:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    uids = split["test"]
    if len(uids) != len(set(uids)):
        raise ValueError("test UIDs are not unique")
    reasons: Counter[str] = Counter()
    lengths: list[int] = []
    valid_x: list[int] = []
    valid_y: list[int] = []
    with zipfile.ZipFile(cad_zip_path) as archive:
        names = set(archive.namelist())
        for uid in uids:
            member = f"cad_seq/{uid}/seq/{uid.split('/')[-1]}.pth"
            if member not in names:
                reasons["missing_zip_member"] += 1
                continue
            try:
                sample = torch.load(
                    io.BytesIO(archive.read(member)),
                    map_location="cpu",
                    weights_only=True,
                )
                vec = sample["vec"]["cad_vec"]
                mask = sample["mask_cad_dict"]["key_padding_mask"]
                if vec.shape != (272, 2) or mask.shape != (272, 2):
                    raise ValueError("unexpected fixed vector or mask shape")
                if mask.dtype != torch.bool or not torch.equal(mask, vec == 0):
                    raise ValueError("official mask differs from vector == 0")
                diag = validate_cad_vec(
                    vec,
                    mask,
                    sample["vec"]["flag_vec"],
                    sample["vec"]["index_vec"],
                )
                lengths.append(diag.n_real_tokens)
                valid_x.append(diag.n_valid_x)
                valid_y.append(diag.n_valid_y)
            except ScorerInputError as exc:
                reasons[exc.reason] += 1
            except Exception as exc:
                reasons[f"other:{type(exc).__name__}:{exc}"] += 1
    return {
        "n_test": len(uids),
        "n_valid": len(lengths),
        "reasons": dict(sorted(reasons.items())),
        "real_token_quantiles": quantiles(lengths),
        "valid_x_quantiles": quantiles(valid_x),
        "valid_y_quantiles": quantiles(valid_y),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", required=True, type=Path)
    parser.add_argument("--cad-zip", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.split_json, args.cad_zip)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["n_valid"] == result["n_test"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
