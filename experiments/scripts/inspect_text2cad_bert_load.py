"""Zero-forward load audit of the pinned Text2CAD BERT and query token lengths."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import BertModel, BertTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[int]) -> dict[str, int]:
    values = sorted(values)
    return {
        str(percent): values[round((len(values) - 1) * percent / 100)]
        for percent in (0, 50, 90, 95, 99, 100)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert-dir", required=True, type=Path)
    parser.add_argument("--query-jsonl", type=Path)
    args = parser.parse_args()

    weight_path = args.bert_dir / "model.safetensors"
    result = {"status": "structure_only_no_forward", "bert_sha256": sha256(weight_path)}
    with safe_open(weight_path, framework="pt", device="cpu") as source:
        keys = list(source.keys())
        result["source_bert_keys"] = sum(key.startswith("bert.") for key in keys)
        result["source_cls_keys"] = sum(key.startswith("cls.") for key in keys)
        source_word_embeddings = source.get_tensor("bert.embeddings.word_embeddings.weight")

    model, loading = BertModel.from_pretrained(
        str(args.bert_dir), local_files_only=True, output_loading_info=True
    )
    result["model_parameter_tensors"] = len(list(model.state_dict()))
    result["missing_keys"] = loading.get("missing_keys", [])
    result["unexpected_keys"] = loading.get("unexpected_keys", [])
    result["mismatched_keys"] = loading.get("mismatched_keys", [])
    result["error_msgs"] = loading.get("error_msgs", [])
    result["word_embeddings_exact_equal"] = bool(
        torch.equal(model.embeddings.word_embeddings.weight.detach().cpu(), source_word_embeddings)
    )
    del model

    if args.query_jsonl is not None:
        tokenizer = BertTokenizer.from_pretrained(str(args.bert_dir), local_files_only=True)
        texts = []
        for line in args.query_jsonl.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            texts.append(row["text"])
        unique = list(dict.fromkeys(texts))
        lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in unique]
        result["n_queries"] = len(texts)
        result["n_unique_texts"] = len(unique)
        result["n_over_512"] = sum(length > 512 for length in lengths)
        result["token_length_quantiles"] = quantiles(lengths)

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
