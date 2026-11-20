import hashlib
import importlib.util
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / "scripts"))


class DuoduoCLIPEncoderTests(unittest.TestCase):
    def encoder(self):
        name = "encode_retrieval_benchmark_duoduoclip"
        self.assertIsNotNone(importlib.util.find_spec(name), "missing DuoduoCLIP encoder")
        return importlib.import_module(name)

    def test_processor_output_preserves_object_and_view_axes(self):
        encoder = self.encoder()

        def processor(*, images, return_tensors):
            self.assertEqual(return_tensors, "np")
            values = np.asarray(images, dtype=np.float32).reshape(-1, 1, 1, 1)
            return {"pixel_values": np.broadcast_to(values, (16, 3, 224, 224))}

        groups = [list(range(8)), list(range(32, 40))]
        pixels = encoder.pack_views(processor, groups, frame_count=8)
        self.assertEqual(pixels.shape, (2, 8, 3, 224, 224))
        np.testing.assert_array_equal(pixels[:, :, 0, 0, 0], groups)
        with self.assertRaisesRegex(ValueError, "eight|8"):
            encoder.pack_views(processor, [groups[0][:-1], groups[1]], frame_count=8)

    def test_model_call_receives_joint_object_tensor(self):
        encoder = self.encoder()

        def processor(*, images, return_tensors):
            colors = np.asarray(images, dtype=np.float32).reshape(-1, 1, 1, 1)
            return {"pixel_values": np.broadcast_to(colors, (len(images), 3, 224, 224))}

        class FakeTorch:
            inference_mode = staticmethod(nullcontext)
            from_numpy = staticmethod(lambda value: value)

        class Result:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class FakeModel:
            def get_image_features(self, *, pixel_values):
                assert pixel_values.shape == (2, 8, 3, 224, 224)
                first_last = pixel_values[:, (0, -1), 0, 0, 0]
                return Result(first_last)

        model = encoder.ModelEncoder(FakeModel(), processor, None, FakeTorch(), dim=2, window=77)
        observed = model.images([list(range(8)), list(range(40, 48))], frame_count=8)
        np.testing.assert_array_equal(observed, [[0, 7], [40, 47]])

    def test_token_prefix_keeps_termination_and_full_length(self):
        encoder = self.encoder()

        class Tokenizer:
            eos_token_id = 49407

            def __call__(self, texts, **kw):
                if not kw["truncation"]:
                    return {"input_ids": [[49406, 8, 9, 10, 49407], [49406, 27, 49407]]}
                return {"input_ids": np.array([[49406, 8, 9, 49407], [49406, 27, 49407, 49407]]),
                        "attention_mask": np.array([[1, 1, 1, 1], [1, 1, 1, 0]])}

        ids, mask, counts = encoder.tokenized_prefix(Tokenizer(), ["long exact text", "short exact text"], 4)
        np.testing.assert_array_equal(ids[:, 2:], [[9, 49407], [49407, 49407]])
        np.testing.assert_array_equal(mask.sum(axis=1), [4, 3])
        self.assertEqual(counts, [5, 3])

        class LostEnd(Tokenizer):
            def __call__(self, texts, **kw):
                result = super().__call__(texts, **kw)
                if kw["truncation"]:
                    result["input_ids"][0, 3] = 10
                return result

        with self.assertRaisesRegex(ValueError, "prefix|EOS"):
            encoder.tokenized_prefix(LostEnd(), ["long exact text", "short exact text"], 4)

    def test_render_and_embedding_failures_keep_historical_positions(self):
        encoder = self.encoder()
        gallery = [{"uid": f"0000/{i:08}", "cad_index": i} for i in (2, 5, 11, 17)]
        records = [row | {"valid": True, "source_sha256": "0" * 64} for row in gallery]
        records[1].update(valid=False, reason="empty render")

        def image_forward(images, frame_count):
            self.assertEqual(frame_count, 8)
            self.assertEqual([shape[0] for shape in images], [2, 11, 17])
            return np.array([[3, 4], [np.nan, 0], [0, -3]], dtype=np.float32)

        vectors, statuses = encoder.gallery_embeddings(gallery, records, lambda rec: [rec["cad_index"]] * 8,
                                                       image_forward, batch_size=4, dim=2)
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0, 0], [0, 0], [0, -1]])
        self.assertEqual(vectors.dtype, np.dtype("<f8"))
        self.assertEqual([r["cad_index"] for r in statuses], [2, 5, 11, 17])
        self.assertEqual([r["valid"] for r in statuses], [True, False, False, True])
        self.assertIn("empty render", statuses[1]["reason"])
        self.assertIn("nonfinite", statuses[2]["reason"])

    def test_text_status_records_window_without_dropping_zero_embedding(self):
        encoder = self.encoder()
        texts = [{"text_index": i, "text": q, "text_sha256": hashlib.sha256(q.encode()).hexdigest()}
                 for i, q in enumerate(["original first text", "original second text"])]

        def text_forward(strings):
            self.assertEqual(strings, [q["text"] for q in texts])
            return np.array([[0, 8], [0, 0]], dtype=np.float32), [91, 6]

        vectors, statuses = encoder.text_embeddings(texts, text_forward, batch_size=2, dim=2, window=77)
        np.testing.assert_array_equal(vectors, [[0, 1], [0, 0]])
        self.assertEqual([r["valid"] for r in statuses], [True, False])
        self.assertEqual([r["truncated"] for r in statuses], [True, False])
        self.assertEqual([r["tokens_before_truncation"] for r in statuses], [91, 6])
        self.assertEqual([r["tokens_encoded"] for r in statuses], [77, 6])
        self.assertEqual([r["text_sha256"] for r in statuses], [q["text_sha256"] for q in texts])

    def test_snapshot_rejects_unpinned_overrides_and_changed_demo_bytes(self):
        encoder = self.encoder()
        payloads = {"model.safetensors": b"weight bytes", "config.json": b"{}",
                    "assets/demo/view_000.png": b"demo image"}
        pins = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
        model_spec = {"repo": "publisher/model", "revision": "1" * 40,
                      "weights_bytes": len(payloads["model.safetensors"]),
                      "files": {name: pins[name] for name in ("model.safetensors", "config.json")},
                      "demo_images": {"assets/demo/view_000.png": pins["assets/demo/view_000.png"]}}
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            for name, data in payloads.items():
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            self.assertEqual(encoder.audit_snapshot(snapshot, model_spec)["files"], pins)
            override = snapshot / "adapter_config.json"
            override.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unpinned"):
                encoder.audit_snapshot(snapshot, model_spec)
            override.unlink()
            (snapshot / "assets/demo/view_000.png").write_bytes(b"changed!")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                encoder.audit_snapshot(snapshot, model_spec)

    def test_published_axis_reuses_complete_bytes_and_rejects_corruption(self):
        encoder = self.encoder()
        rows = [{"uid": "0000/00000007", "cad_index": 7}]
        meta = {"method": "duoduoclip", "axis_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}

        def build():
            return np.array([[1, 0]], dtype="<f8"), [rows[0] | {"valid": True}]

        def do_not_rebuild():
            self.fail("complete artifact should be reused")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            encoder.publish_axis(output, "gallery", rows, meta, 2, build)
            encoder.publish_axis(output, "gallery", rows, meta, 2, do_not_rebuild)
            with (output / "gallery_status.jsonl").open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                encoder.publish_axis(output, "gallery", rows, meta, 2, do_not_rebuild)


if __name__ == "__main__":
    unittest.main()
