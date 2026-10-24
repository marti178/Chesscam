"""Convierte ChessSight synthetic (HF tchauffi/chesssight-synthetic-40k, CC BY-NC 4.0) al formato del contrato.

Renders de Blender con esquinas exactas (`corners_px`, horario desde a8: a8, h8, h1, a1) y la grilla
(64 enteros desde a8, 0 vacía, 1–6 PNBRQK, 7–12 pnbrqk: el mismo orden que CONTRACT.md).

  python -m chessscan.prepare_chesssight --out data/prepared_chesssight --max-train 8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g
from .prepare_chessred import A1_AT


def convert(r):
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), 1), cv2.COLOR_BGR2RGB)
    board = np.array(r["corners_px"], np.float64)
    tl = g.sort_corners(board)[0]
    start = int(np.argmin(np.linalg.norm(board - tl, axis=1)))
    a1 = A1_AT[(3 - start) % 4]
    H = g.homography(np.roll(board, -start, axis=0))
    grid = r["grid"]
    by_sq = {"abcdefgh"[k % 8] + str(8 - k // 8): grid[k] for k in range(64)}
    lab = np.array([by_sq[g.grid_to_square(i // 8, i % 8, a1)] for i in range(64)], np.int64)
    return g.warp(img, H), lab, a1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/chesssight"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-val", type=int, default=600)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    for split, pat, mx in (("train", "train-*.parquet", a.max_train), ("val", "validation-*.parquet", a.max_val)):
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        canon = np.lib.format.open_memmap(d / "canon.npy", "w+", np.uint8, (mx, g.CH, g.CW, 3))
        labels, names, n = [], [], 0
        bar = tqdm(total=mx, desc=split)
        for f in sorted(a.root.glob(pat)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["image", "id", "grid", "corners_px"]):
                for r in b.to_pylist():
                    if n >= mx:
                        break
                    cv, lab, a1 = convert(r)
                    canon[n] = cv
                    labels.append(lab)
                    names.append(f"{r['id']} a1={a1}")
                    n += 1
                    bar.update()
        canon.flush()
        assert n == mx, (split, n)
        np.save(d / "labels.npy", np.stack(labels))
        (d / "names.txt").write_text("\n".join(names) + "\n")
        print(split, n, flush=True)


if __name__ == "__main__":
    main()
