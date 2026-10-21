"""Evaluación de un .tflite sobre datos preparados y latencia en CPU de escritorio.

  python -m chessscan.evaluate --model export/chess_square_classifier.tflite \
      --data data/prepared/test data/prepared_real/test --bench --out runs/v1/eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from .data import Boards, all_crops
from .pipeline import metrics, tflite_predict


def bench(model: Path, threads: int, runs: int = 50) -> dict:
    it = tf.lite.Interpreter(model_path=str(model), num_threads=threads)
    it.allocate_tensors()
    x = np.random.default_rng(0).integers(0, 256, (64, 128, 64, 3)).astype(np.float32)
    it.set_tensor(it.get_input_details()[0]["index"], x)
    for _ in range(5):
        it.invoke()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        it.invoke()
        ts.append((time.perf_counter() - t) * 1000)
    return {"threads": threads, "batch": 64, "median_ms": float(np.median(ts)),
            "p90_ms": float(np.percentile(ts, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, nargs="+", default=[])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = {"model": a.model.name, "bytes": a.model.stat().st_size}
    for d in a.data:
        x, y = all_crops(Boards.load(d))
        m = metrics(tflite_predict(a.model, x), y)
        res[str(d)] = m
        print(f"{d}: casillas {m['square_acc']:.4f}  ocupadas {m['occupied_acc']:.4f}  "
              f"vacías {m['empty_acc']:.4f}  tableros perfectos {m['board_acc_all64']:.3f}  "
              f"errores/tablero {m['mean_errors_per_board']:.2f}")
    if a.bench:
        res["cpu"] = platform.processor()
        res["latency"] = [bench(a.model, t) for t in (1, 4)]
        for b in res["latency"]:
            print(f"latencia {b['threads']} hilos: mediana {b['median_ms']:.1f} ms, p90 {b['p90_ms']:.1f} ms (lote 64)")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
