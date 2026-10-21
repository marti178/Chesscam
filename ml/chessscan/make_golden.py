"""Genera los vectores dorados de ml/golden/ (ver CONTRACT §6).

Casos (todos salen de un render del split val de chesscog, CC-BY 4.0):
  full_1200x800    imagen original, esquinas enteras
  small_640x427    reducida a ~640 px (como el análisis de CameraX), esquinas no enteras
  cut_corner       recortada para que una esquina quede fuera del fotograma (bordes negros)

Uso:  python -m chessscan.make_golden --render ml/data/raw/extracted/val/0418.png \
          --model ml/export/chess_square_classifier.tflite --out ml/golden
Sin --model solo escribe la parte geométrica.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g
from .prepare_chesscog import load_render


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def run_tflite(model: Path, x: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(model), num_threads=1)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    return interp.get_tensor(interp.get_output_details()[0]["index"]).copy()


def write_case(out: Path, name: str, img: np.ndarray, corners: np.ndarray, labels: np.ndarray,
               fen: str, a1: str, model: Path | None) -> None:
    d = out / name
    if d.exists():
        shutil.rmtree(d)
    H = g.homography(corners)
    canon = g.warp(img, H)
    crops = g.crop_all(canon)
    x = g.model_input(crops)
    write_png(d / "image.png", img)
    write_png(d / "canonical.png", canon)
    for i, c in enumerate(crops):
        write_png(d / "crops" / f"r{i // 8}c{i % 8}.png", c)
    crops.tofile(d / "input_u8.bin")  # el tensor float32 es exactamente estos bytes convertidos
    meta = {
        "contract_version": g.CONTRACT_VERSION,
        "image_size_wh": [img.shape[1], img.shape[0]],
        "corners_TL_TR_BR_BL": corners.tolist(),
        "H_row_major": H.tolist(),
        "constants": {"S": g.S, "MX": g.MX, "MY": g.MY, "CW": g.CW, "CH": g.CH,
                      "CROP_H": g.CROP_H, "CROP_W": g.CROP_W},
        "row_bounds_y0_y1": [list(g.row_bounds(r)) for r in range(8)],
        "col_bounds_x0_x1": [list(g.col_bounds(c)) for c in range(8)],
        "classes": g.CLASSES,
        "fen": fen,
        "a1_corner": a1,
        "true_labels": [g.CLASSES[i] for i in labels],
    }
    if model is not None:
        logits = run_tflite(model, x)
        logits.astype("<f4").tofile(d / "output_logits_f32.bin")
        meta["model_file"] = model.name
        meta["expected_top1"] = [g.CLASSES[i] for i in logits.argmax(1)]
        meta["expected_top1_matches_truth"] = int((logits.argmax(1) == labels).sum())
    (d / "case.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path, default=Path("ml/golden"))
    a = ap.parse_args()

    meta = json.loads(a.render.with_suffix(".json").read_text())
    fen, a1 = meta["fen"], ("BL" if meta["white_turn"] else "TR")
    img, corners, labels = load_render(a.render)

    write_case(a.out, "full_1200x800", img, corners, labels, fen, a1, a.model)

    sx, sy = 640 / img.shape[1], 427 / img.shape[0]
    small = cv2.resize(img, (640, 427), interpolation=cv2.INTER_AREA)
    write_case(a.out, "small_640x427", small, corners * [sx, sy], labels, fen, a1, a.model)

    x0, y0 = 300, 0  # BL queda en x < 0
    cut = np.ascontiguousarray(img[y0:, x0:])
    write_case(a.out, "cut_corner", cut, corners - [x0, y0], labels, fen, a1, a.model)


if __name__ == "__main__":
    main()
