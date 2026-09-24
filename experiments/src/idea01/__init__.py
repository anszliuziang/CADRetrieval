"""Idea 01 P0: frozen Text-to-CadQuery Qwen2.5-3B-SFT as a conditional scorer.

Package contents:
- prompts:    fixed instruction prefix (official Text-to-CadQuery format)
- scoring:    numpy teacher-forced log-prob core (testable without torch)
- data_audit: JSONL <-> CSV <-> CadQuery.zip source/UID verification
- coverage:   tokenizer-based length / boundary / boilerplate coverage
- selection:  deterministic P0 candidate + mismatch-query selection

Task contract: experiments/plans/IDEA01_P0_IMPLEMENTATION_2026-09-24.md
"""
