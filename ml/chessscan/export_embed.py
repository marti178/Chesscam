"""Exporta un clasificador con una segunda salida de features (contrato v1.1, CONTRACT.md §6).

Salidas: `logits` [64,13] (igual que v1) y `embedding` [64,D]: el vector después del
GlobalAveragePooling y antes de dropout/densa, **normalizado L2**. D = 576 (Small) o 960 (Large).

  python -m chessscan.export_embed --model runs/v5/model_calibrated.keras --out export/chess_square_classifier_emb.tflite
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import keras
import tensorflow as tf

from . import geometry as g


def with_embedding(model: keras.Model) -> keras.Model:
    gap = next(l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D))
    feats = gap.output
    emb = keras.layers.UnitNormalization(axis=-1, name="embedding")(feats)
    return keras.Model(model.input, {"logits": model.output, "embedding": emb})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    m = with_embedding(keras.models.load_model(a.model))
    sm = Path(tempfile.mkdtemp()) / "sm"
    arch = keras.export.ExportArchive()
    arch.track(m)
    arch.add_endpoint(name="serve", fn=lambda x: m(x, training=False),
                      input_signature=[tf.TensorSpec([64, g.CROP_H, g.CROP_W, 3], tf.float32, name="crops")])
    arch.write_out(str(sm))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm), signature_keys=["serve"])
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    a.out.write_bytes(conv.convert())
    shutil.rmtree(sm.parent, ignore_errors=True)
    print("ok", a.out, a.out.stat().st_size)


if __name__ == "__main__":
    main()
