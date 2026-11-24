import numpy as np
import pytest

from experiments.scripts import replay_910a_training_token_cache as replay


def test_replay_recipe_allows_padding_and_probe_validation_only():
    original = dict(code_sha=replay.SOURCE_TRAINER_SHA, script_sha256="old", smoke_validation_count=0,
                    world=2, seed=17, optimizer="AdamW", global_batch=16)
    diagnostic = dict(original, code_sha="diagnostic", script_sha256="diagnostic-file",
                      smoke_validation_count=replay.VALIDATION_ROWS, training_text_pad_multiple=64)
    replay.compare_recipes(original, diagnostic)
    with pytest.raises(ValueError, match="original training contract"):
        replay.compare_recipes(original, dict(diagnostic, global_batch=32))


def test_cache_reader_requires_complete_nonoverlapping_spans(tmp_path):
    features = tmp_path / "effective_token_features.npy"
    index = tmp_path / "index.json"
    np.save(features, np.zeros((3, 1024), dtype=np.float32))
    receipt = dict(complete=True, source=replay.SOURCE, epoch=1, start_update=replay.START_UPDATE,
                   stop_update=replay.STOP_UPDATE, pad_multiple=64, feature_shape=[3, 1024], distinct_texts=2,
                   bert_sha256=replay.native.PINNED_BERT_FILE_SHA256, feature_sha256=replay.native.sha256_file(features))
    for offset in (1, 0):
        replay.native.write_json(index, {"a": dict(offset=0, length=1), "b": dict(offset=offset, length=2)})
        replay.native.write_json(tmp_path / "receipt.json", dict(receipt, index_sha256=replay.native.sha256_file(index)))
        if offset == 1:
            _, recovered, _ = replay.cache_reader(tmp_path)
            assert recovered.shape == (3, 1024)
        else:
            with pytest.raises(ValueError, match="index span"):
                replay.cache_reader(tmp_path)
