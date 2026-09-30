"""OCC-free, explicit-device port of the frozen Text2CAD forward architecture.

Upstream: Text2CAD (Mohammad Sadil Khan et al., NeurIPS 2024, DFKI GmbH),
read-only mirror at Git SHA 669a81472ede2269837c22a4d00070eaac360c82,
license CC BY-NC-SA 4.0 (see ``LICENSE`` in this directory). The ported
modules are adaptations and remain under CC BY-NC-SA 4.0; ``NOTICE.md``
enumerates every deviation from the upstream files.

Contents:
  * ``model``       -- torch-only architecture: masks, embedders, attention,
                       decoder, adaptive layer, Text2CAD shell, and the
                       teacher-forcing slicing glue (no transformers import);
  * ``text_embed``  -- adapted TextEmbedder (explicit device, pinned local
                       BERT, >512-token rejection instead of silent truncation);
  * ``loader``      -- strict asset/checkpoint/BERT loader with SHA, key-set,
                       shape, zero-tensor and numeric gates (no strict=False,
                       no forward).

Stage B1 scope: no real model forward is run anywhere in this package.
"""
