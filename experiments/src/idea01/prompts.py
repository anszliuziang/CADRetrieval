"""Fixed prompt format, matching official Text-to-CadQuery training/inference.

Official source (pinned): Text-to-CadQuery/Text-to-CadQuery@4f7f50176f3c642f6d897c4a8670f7a83acb31bd
- train/train_qwen-3B.py and inference/step1_generate_CadQuery/inference_qwen-3B.py
  both build "### Instruction:\n{input}\n\n### Response:\n" + output, with the
  tokenizer EOS appended to the full training string.
- Training kept only full sequences of <=1024 tokens.

The conditional score in this project sums log-probs of response tokens + EOS
only; the prefix is conditioning context, never a target.
"""

PREFIX_TEMPLATE = "### Instruction:\n{input}\n\n### Response:\n"


def build_prefix(query: str) -> str:
    """Instruction prefix for a query text. Query is inserted verbatim."""
    return PREFIX_TEMPLATE.replace("{input}", query)


def build_full_text(query: str, output: str) -> str:
    """Full string = prefix + raw program text. EOS is appended at token level,
    not as text, so the string boundary stays exactly len(prefix) characters."""
    return build_prefix(query) + output
