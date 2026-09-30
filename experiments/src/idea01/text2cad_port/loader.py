"""Strict, zero-forward-friendly loader for the pinned frozen Text2CAD assets.

Task contract: experiments/plans/IDEA01_M1C_TEXT2CAD_SCORE_BENCH_2026-09-30.md
(G0.2/G0.4) and experiments/reviews/IDEA01_M1C_STAGE_A_CODEX_REVIEW_2026-09-30.md
(Stage B gates). This loader is the ONLY sanctioned way to assemble the frozen
model; it refuses -- with stable reason codes -- any of:

  * asset SHA256 mismatch (checkpoint file, five pinned BERT files);
  * checkpoint structure drift: top-level keys other than
    ``{"model_state_dict"}``, keys without the official DataParallel
    ``module.`` prefix, or any ``base_text_embedder.*`` (BERT) tensor inside
    the checkpoint (BERT must come from the pinned base weights);
  * any non-BERT missing key, unexpected key or shape mismatch against BOTH
    the freshly built ported architecture and (when given) the independent
    expected-keys manifest ``experiments/configs/text2cad_checkpoint_expected_keys_20260930.json``;
  * any unexpected all-zero tensor, or an expected-zero tensor that is not
    zero (see :data:`EXPECTED_ZERO_TENSOR_KEYS`);
  * numeric inequality between loaded parameters and checkpoint tensors;
  * BERT loading anomalies: missing/mismatched/error keys, unexpected keys
    outside the unused ``cls.*`` MLM head, word embeddings not exactly equal
    to the pinned safetensors tensor, or ``max_position_embeddings != 512``.

``strict=False`` is never used. Loading is STRUCTURE ONLY -- no forward is
run anywhere in this module (the current stage's real-model forward budget
is 0), so it doubles as the zero-forward checkpoint-loading diagnostic
required by ACTIVE_TASK.md.

License note: the loader itself is project glue, but it assembles the ported
architecture from ``model.py`` / ``text_embed.py``, which are CC BY-NC-SA 4.0
adaptations of upstream Text2CAD (Git SHA
669a81472ede2269837c22a4d00070eaac360c82); see NOTICE.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .model import (
    CAD_CLASS_INFO,
    MAX_CAD_SEQUENCE_LENGTH,
    OFFICIAL_INFERENCE_CONFIG,
    OFFICIAL_SOURCE_GIT_SHA,
    CADDecoder,
    Text2CAD,
    TextAdaptiveLayer,
)
from .text_embed import TextEmbedder

# ---------------------------------------------------------------------------
# Pinned asset fingerprints (experiments/configs/text2cad_original_asset_sha256_20260930.json)
# ---------------------------------------------------------------------------

#: Original-author Text2CAD_1.0.pth (91,601,866 bytes), per ACTIVE_TASK.md.
PINNED_CHECKPOINT_SHA256 = "235b137e72f97c740550e29698c342967d3be7c1d4cd0067a1b3380389034598"

#: google-bert/bert-large-uncased @ 6da4b6a26a1877e173fca3225479512db81a5e5b.
PINNED_BERT_FILE_SHA256 = {
    "config.json": "f9a38c9ffa369f288c3206d0bf352dfed11d7bdee93aa5b2de6d7fdc7bff7871",
    "model.safetensors": "28349a5c57338d0dca591bb95de533897c2399095c69d61e09604b46f444a4c4",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}

#: The checkpoint's 172 non-BERT tensors include exactly two all-zero tensors
#: (170 non-zero), established by the zero-forward structure audit
#: (experiments/runs/m1c_structure_20260930/ and the local full-key audit).
#: They are the ``norm_3`` biases of the two decoder layers WITHOUT
#: cross-attention (ca_level_start=2): CADDecoderLayer creates norm_3 for
#: every layer but only applies it inside ``if self.use_ca:``
#: (decoder.py:396-412 upstream), so those biases never received gradients
#: and remain at their LayerNorm init value 0 (the matching weights remain at
#: init value 1 -- verified all-ones, unlike the trained layers' ~0.135 mean).
EXPECTED_ZERO_TENSOR_KEYS = frozenset(
    {
        "cad_decoder.cad_decoder_layers.0.norm_seq.norm_3.bias",
        "cad_decoder.cad_decoder_layers.1.norm_seq.norm_3.bias",
    }
)

#: Pinned BERT base tensor counts (zero-forward server audit,
#: experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md).
EXPECTED_BERT_PARAMETER_TENSORS = 391
EXPECTED_BERT_SOURCE_TENSORS = 391
#: The pinned model.safetensors belongs to BertForMaskedLM; loading it as
#: BertModel leaves exactly the MLM head unused.
ALLOWED_BERT_UNEXPECTED_PREFIX = "cls."
EXPECTED_MAX_POSITION_EMBEDDINGS = 512

# ---------------------------------------------------------------------------
# Rejection reason codes (stable strings; asserted in tests and ledgers)
# ---------------------------------------------------------------------------

REASON_SHA_MISMATCH = "asset_sha256_mismatch"
REASON_ASSET_MISSING = "asset_missing"
REASON_BAD_CHECKPOINT_STRUCTURE = "bad_checkpoint_structure"
REASON_BERT_TENSOR_IN_CHECKPOINT = "bert_tensor_in_checkpoint"
REASON_MISSING_KEYS = "missing_keys"
REASON_UNEXPECTED_KEYS = "unexpected_keys"
REASON_SHAPE_MISMATCH = "shape_mismatch"
REASON_UNEXPECTED_ZERO_TENSOR = "unexpected_zero_tensor"
REASON_EXPECTED_ZERO_VIOLATION = "expected_zero_violation"
REASON_NUMERIC_MISMATCH = "numeric_mismatch"
REASON_MANIFEST_MISMATCH = "expected_keys_manifest_mismatch"
REASON_BERT_LOAD_ANOMALY = "bert_load_anomaly"
REASON_BERT_CONFIG_MISMATCH = "bert_config_mismatch"


class AssetVerificationError(ValueError):
    """Raised when a pinned asset or a strict load violates the contract.

    ``reason`` is one of the ``REASON_*`` constants so callers can ledger
    rejections without parsing messages.
    """

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path) -> str:
    """Chunked SHA256 of one file (4 MiB blocks; no full-file read)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_sha256(path: Path, expected_sha256: str, label: str) -> dict:
    if not path.is_file():
        raise AssetVerificationError(
            REASON_ASSET_MISSING, f"{label}: no such file: {path}"
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AssetVerificationError(
            REASON_SHA_MISMATCH,
            f"{label}: sha256 {actual} != pinned {expected_sha256}",
        )
    return {"path": str(path), "sha256": actual}


# ---------------------------------------------------------------------------
# Checkpoint verification and strict loading (torch-only; NO forward)
# ---------------------------------------------------------------------------


def load_verified_checkpoint(
    checkpoint_path,
    expected_sha256: str = PINNED_CHECKPOINT_SHA256,
) -> "tuple[dict, dict]":
    """Verify SHA + structure of the official checkpoint and return its state.

    Returns ``(state, report)`` where ``state`` maps the 172 non-BERT keys
    (``module.`` prefix stripped) to CPU tensors and ``report`` carries the
    structural facts for the run ledger. ``torch.load`` runs with
    ``weights_only=True``; no model is built and no forward is executed here.
    """
    checkpoint_path = Path(checkpoint_path)
    report = {
        "checkpoint": _verify_file_sha256(
            checkpoint_path, expected_sha256, "Text2CAD_1.0.pth"
        ),
    }
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or set(blob.keys()) != {"model_state_dict"}:
        raise AssetVerificationError(
            REASON_BAD_CHECKPOINT_STRUCTURE,
            f"top-level keys must be exactly {{'model_state_dict'}}, "
            f"got {sorted(blob.keys()) if isinstance(blob, dict) else type(blob).__name__}",
        )
    raw_state = blob["model_state_dict"]
    if not isinstance(raw_state, dict) or not raw_state:
        raise AssetVerificationError(
            REASON_BAD_CHECKPOINT_STRUCTURE, "model_state_dict is empty or not a dict"
        )
    unprefixed = [k for k in raw_state if not k.startswith("module.")]
    if unprefixed:
        raise AssetVerificationError(
            REASON_BAD_CHECKPOINT_STRUCTURE,
            f"{len(unprefixed)} tensor(s) lack the official 'module.' prefix, "
            f"e.g. {unprefixed[:5]}",
        )
    state = {}
    for key, value in raw_state.items():
        stripped = key[len("module."):]
        if stripped.startswith("base_text_embedder."):
            raise AssetVerificationError(
                REASON_BERT_TENSOR_IN_CHECKPOINT,
                f"checkpoint unexpectedly carries BERT tensor {key}; BERT must "
                "come from the pinned bert-large-uncased base weights",
            )
        if not isinstance(value, torch.Tensor):
            raise AssetVerificationError(
                REASON_BAD_CHECKPOINT_STRUCTURE,
                f"{key}: expected torch.Tensor, got {type(value).__name__}",
            )
        state[stripped] = value
    zero_keys = sorted(
        key for key, value in state.items() if int(torch.count_nonzero(value).item()) == 0
    )
    report.update(
        {
            "n_tensors": len(state),
            "n_zero_tensors": len(zero_keys),
            "zero_tensor_keys": zero_keys,
            "dtypes": sorted({str(v.dtype) for v in state.values()}),
        }
    )
    return state, report


def load_expected_keys_manifest(manifest_path) -> dict:
    """Load the independent expected-keys manifest (configs/...json)."""
    with Path(manifest_path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    for field in ("tensors", "zero_tensor_keys", "n_tensors"):
        if field not in manifest:
            raise AssetVerificationError(
                REASON_MANIFEST_MISMATCH, f"manifest lacks field {field!r}"
            )
    return manifest


def nonbert_state_dict_keys(model: nn.Module) -> "set[str]":
    """State-dict keys of the ported model EXCLUDING the frozen BERT subtree
    (mirrors text2cad.py:123-127 get_trainable_state_dict)."""
    return {
        key
        for key in model.state_dict()
        if "base_text_embedder" not in key.split(".")
    }


def strict_load_frozen_weights(
    model: Text2CAD,
    state: dict,
    expected_zero_keys=EXPECTED_ZERO_TENSOR_KEYS,
    expected_keys_manifest=None,
) -> dict:
    """Strictly load the verified checkpoint state into the ported model.

    Order of gates (all must pass BEFORE any parameter is touched):

      1. key-set equality between the checkpoint and the model's non-BERT
         parameters (missing / unexpected are fatal, listed explicitly);
      2. per-key shape equality;
      3. when ``expected_keys_manifest`` (path or parsed dict) is given,
         three-way equality of key sets and shapes against that independent
         manifest (guards against a drift in BOTH model and checkpoint);
      4. zero-tensor policy: the set of all-zero checkpoint tensors must
         equal ``expected_zero_keys`` exactly -- unexpected zeros are
         rejected, and so are expected-zero keys that are not zero;
      5. per-submodule ``load_state_dict(..., strict=True)`` (never
         ``strict=False`` on the whole model);
      6. exact numeric equality between every loaded parameter and the
         checkpoint tensor (``torch.equal``).

    Returns a report dict for the run ledger. NO forward is executed.
    """
    if isinstance(expected_keys_manifest, (str, Path)):
        expected_keys_manifest = load_expected_keys_manifest(expected_keys_manifest)

    model_keys = nonbert_state_dict_keys(model)
    checkpoint_keys = set(state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing:
        raise AssetVerificationError(
            REASON_MISSING_KEYS,
            f"{len(missing)} non-BERT parameter(s) have no checkpoint tensor, "
            f"e.g. {missing[:5]}",
        )
    if unexpected:
        raise AssetVerificationError(
            REASON_UNEXPECTED_KEYS,
            f"{len(unexpected)} checkpoint tensor(s) have no model parameter, "
            f"e.g. {unexpected[:5]}",
        )

    model_state = model.state_dict()
    shape_mismatch = sorted(
        key
        for key in checkpoint_keys
        if tuple(model_state[key].shape) != tuple(state[key].shape)
    )
    if shape_mismatch:
        raise AssetVerificationError(
            REASON_SHAPE_MISMATCH,
            "; ".join(
                f"{key}: model {tuple(model_state[key].shape)} vs "
                f"checkpoint {tuple(state[key].shape)}"
                for key in shape_mismatch[:5]
            ),
        )

    if expected_keys_manifest is not None:
        manifest_tensors = expected_keys_manifest["tensors"]
        manifest_keys = set(manifest_tensors)
        manifest_mismatch = []
        if manifest_keys != checkpoint_keys:
            manifest_mismatch.append(
                f"key sets differ (manifest-only: "
                f"{sorted(manifest_keys - checkpoint_keys)[:5]}, "
                f"checkpoint-only: {sorted(checkpoint_keys - manifest_keys)[:5]})"
            )
        else:
            for key in sorted(manifest_keys):
                entry = manifest_tensors[key]
                if list(entry["shape"]) != list(state[key].shape):
                    manifest_mismatch.append(
                        f"{key}: manifest {entry['shape']} vs checkpoint "
                        f"{list(state[key].shape)}"
                    )
                if entry["dtype"] != str(state[key].dtype):
                    manifest_mismatch.append(
                        f"{key}: manifest dtype {entry['dtype']} vs "
                        f"checkpoint {state[key].dtype}"
                    )
        if manifest_mismatch:
            raise AssetVerificationError(
                REASON_MANIFEST_MISMATCH, "; ".join(manifest_mismatch[:5])
            )

    zero_keys = {
        key for key, value in state.items() if int(torch.count_nonzero(value).item()) == 0
    }
    unexpected_zero = sorted(zero_keys - set(expected_zero_keys))
    if unexpected_zero:
        raise AssetVerificationError(
            REASON_UNEXPECTED_ZERO_TENSOR,
            f"all-zero checkpoint tensor(s) outside the documented expected "
            f"set: {unexpected_zero}",
        )
    missing_zero = sorted(set(expected_zero_keys) - zero_keys)
    if missing_zero:
        raise AssetVerificationError(
            REASON_EXPECTED_ZERO_VIOLATION,
            f"documented expected-zero tensor(s) are not all-zero: {missing_zero}",
        )

    # Gates passed: strict per-submodule load (BERT subtree untouched).
    adaptive_state = {
        key[len("adaptive_layer."):]: value
        for key, value in state.items()
        if key.startswith("adaptive_layer.")
    }
    decoder_state = {
        key[len("cad_decoder."):]: value
        for key, value in state.items()
        if key.startswith("cad_decoder.")
    }
    if len(adaptive_state) + len(decoder_state) != len(state):
        raise AssetVerificationError(
            REASON_UNEXPECTED_KEYS,
            "checkpoint keys outside adaptive_layer.*/cad_decoder.* namespaces",
        )
    model.adaptive_layer.load_state_dict(adaptive_state, strict=True)
    model.cad_decoder.load_state_dict(decoder_state, strict=True)

    post = model.state_dict()
    numeric_mismatch = sorted(
        key
        for key in checkpoint_keys
        if not torch.equal(post[key].detach().cpu(), state[key].cpu())
    )
    if numeric_mismatch:
        raise AssetVerificationError(
            REASON_NUMERIC_MISMATCH,
            f"loaded parameter(s) differ from checkpoint tensors: "
            f"{numeric_mismatch[:5]}",
        )

    return {
        "n_loaded_tensors": len(state),
        "n_missing_keys": 0,
        "n_unexpected_keys": 0,
        "zero_tensor_keys": sorted(zero_keys),
        "numeric_equality": True,
        "strict": True,
    }


# ---------------------------------------------------------------------------
# BERT verification (needs transformers + safetensors; NO forward)
# ---------------------------------------------------------------------------


def verify_bert_asset_shas(bert_dir, expected_shas: dict = None) -> dict:
    """Verify the five pinned BERT files exist and match their SHA256."""
    if expected_shas is None:
        expected_shas = PINNED_BERT_FILE_SHA256
    bert_dir = Path(bert_dir)
    report = {}
    for filename, sha in sorted(expected_shas.items()):
        report[filename] = _verify_file_sha256(bert_dir / filename, sha, f"BERT {filename}")
    return report


def verify_bert_model_load(embedder: TextEmbedder, bert_dir) -> dict:
    """Gate the already-loaded pinned BERT of ``embedder`` (no forward).

    Mirrors the zero-forward server audit recorded in
    experiments/assets/TEXT2CAD_ORIGINAL_M1C_ASSET_STATUS_2026-09-30.md:
    391/391 base tensors, missing/mismatched/error keys empty, only ``cls.*``
    unexpected, word embeddings exactly equal to the safetensors source.
    """
    from safetensors import safe_open  # lazy: torch-only environments skip
    # this function entirely (the gates then run on the server instead).

    bert_dir = Path(bert_dir)
    loading_info = embedder.loading_info or {}
    problems = []
    if loading_info.get("missing_keys"):
        problems.append(f"missing_keys={loading_info['missing_keys'][:5]}")
    if loading_info.get("mismatched_keys"):
        problems.append(f"mismatched_keys={loading_info['mismatched_keys'][:5]}")
    if loading_info.get("error_msgs"):
        problems.append(f"error_msgs={loading_info['error_msgs'][:2]}")
    unexpected = loading_info.get("unexpected_keys", [])
    bad_unexpected = [
        key for key in unexpected if not key.startswith(ALLOWED_BERT_UNEXPECTED_PREFIX)
    ]
    if bad_unexpected:
        problems.append(f"unexpected keys outside cls.*: {bad_unexpected[:5]}")

    model = embedder.model
    n_parameters = len(list(model.state_dict()))
    if n_parameters != EXPECTED_BERT_PARAMETER_TENSORS:
        problems.append(
            f"BertModel holds {n_parameters} tensors, expected {EXPECTED_BERT_PARAMETER_TENSORS}"
        )
    with safe_open(str(bert_dir / "model.safetensors"), framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        n_source_bert = sum(key.startswith("bert.") for key in source_keys)
        n_source_cls = sum(key.startswith("cls.") for key in source_keys)
        source_word_embeddings = source.get_tensor("bert.embeddings.word_embeddings.weight")
    if n_source_bert != EXPECTED_BERT_SOURCE_TENSORS:
        problems.append(
            f"safetensors carries {n_source_bert} bert.* tensors, expected "
            f"{EXPECTED_BERT_SOURCE_TENSORS}"
        )
    word_embeddings_equal = bool(
        torch.equal(
            model.embeddings.word_embeddings.weight.detach().cpu(),
            source_word_embeddings,
        )
    )
    if not word_embeddings_equal:
        problems.append("word embeddings differ from the pinned safetensors tensor")

    max_position = int(getattr(model.config, "max_position_embeddings", -1))
    if max_position != EXPECTED_MAX_POSITION_EMBEDDINGS:
        raise AssetVerificationError(
            REASON_BERT_CONFIG_MISMATCH,
            f"max_position_embeddings={max_position}, expected "
            f"{EXPECTED_MAX_POSITION_EMBEDDINGS}",
        )
    if problems:
        raise AssetVerificationError(REASON_BERT_LOAD_ANOMALY, "; ".join(problems))
    return {
        "n_parameter_tensors": n_parameters,
        "n_source_bert_tensors": n_source_bert,
        "n_source_cls_tensors": n_source_cls,
        "missing_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
        "unexpected_keys": sorted(unexpected),
        "word_embeddings_exact_equal": word_embeddings_equal,
        "max_position_embeddings": max_position,
    }


# ---------------------------------------------------------------------------
# Full assembly
# ---------------------------------------------------------------------------


def build_text2cad(text_embedder, device, config: dict = None) -> Text2CAD:
    """Build the ported Text2CAD shell around ``text_embedder`` with the
    pinned official inference configuration (weights stay at init until the
    strict loader runs). ``device`` is explicit; nothing here picks CUDA."""
    if config is None:
        config = OFFICIAL_INFERENCE_CONFIG
    adaptive_layer = TextAdaptiveLayer.from_config(config["text_encoder"]["adaptive_layer"])
    cad_decoder = CADDecoder.from_config(config["cad_decoder"], device=device)
    return Text2CAD(
        base_text_embedder=text_embedder,
        adaptive_layer=adaptive_layer,
        cad_decoder=cad_decoder,
        cad_seq_len=config["cad_decoder"]["cad_seq_len"],
    )


def load_frozen_text2cad(
    checkpoint_path,
    bert_dir,
    device,
    expected_checkpoint_sha256: str = PINNED_CHECKPOINT_SHA256,
    expected_bert_shas: dict = None,
    expected_keys_manifest=None,
) -> "tuple[Text2CAD, dict]":
    """Assemble the frozen, strictly loaded Text2CAD for scoring.

    Steps: verify all asset SHAs -> load pinned BERT (local_files_only) and
    gate its loading info -> build the ported architecture -> strict-load the
    172 non-BERT tensors with all key/shape/zero/numeric gates -> move to the
    explicit ``device``, ``eval()``, freeze every parameter (BERT was frozen
    in training, train.py:86-87; the decoder/adaptive layers are frozen for
    evaluation -- requires_grad flags do not affect forward math).

    Returns ``(model, report)``. NO forward is executed anywhere here.
    """
    device = torch.device(device)
    report = {
        "official_source_git_sha": OFFICIAL_SOURCE_GIT_SHA,
        "cad_class_info": CAD_CLASS_INFO,
        "max_cad_sequence_length": MAX_CAD_SEQUENCE_LENGTH,
        "device": str(device),
        "bert_files": verify_bert_asset_shas(bert_dir, expected_bert_shas),
    }
    text_embedder = TextEmbedder.from_local(
        bert_dir,
        max_seq_len=OFFICIAL_INFERENCE_CONFIG["text_encoder"]["text_embedder"]["max_seq_len"],
        device=device,
    )
    report["bert_load"] = verify_bert_model_load(text_embedder, bert_dir)

    state, checkpoint_report = load_verified_checkpoint(
        checkpoint_path, expected_checkpoint_sha256
    )
    report["checkpoint"] = checkpoint_report

    model = build_text2cad(text_embedder, device=device)
    report["strict_load"] = strict_load_frozen_weights(
        model, state, expected_keys_manifest=expected_keys_manifest
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    report["n_nonbert_tensors"] = len(nonbert_state_dict_keys(model))
    report["eval_mode"] = not model.training
    report["all_parameters_frozen"] = all(not p.requires_grad for p in model.parameters())
    return model, report
