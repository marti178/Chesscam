"""Convierte el dataset sintético de chesscog (OSF xf3ka) al formato del contrato.

Por cada split guarda en ml/data/prepared/<split>/:
  canon.npy   uint8 [N, CH, CW, 3]  imágenes canónicas (RGB) según CONTRACT §2
  labels.npy  int64 [N, 64]         clase por casilla (orden fila*8+col)
  names.txt   nombre del render de cada fila

Los recortes se sacan de la imagen canónica en el entrenamiento (así se puede
aumentar la geometría antes de recortar).

Uso:  python -m chessscan.prepare_chesscog --raw ml/data/raw/extracted --out ml/data/prepared
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g


def load_render(png: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = json.loads(png.with_suffix(".json").read_text())
    img = cv2.cvtColor(cv2.imread(str(png), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    corners = g.sort_corners(np.array(meta["corners"], dtype=np.float64))
    # chesscog renderiza desde el lado del jugador al que le toca mover.
    a1 = "BL" if meta["white_turn"] else "TR"
    return img, corners, g.board_labels(meta["fen"], a1)


def convert_split(src: Path, dst: Path) -> None:
    pngs = sorted(src.glob("*.png"))
    dst.mkdir(parents=True, exist_ok=True)
    canon = np.lib.format.open_memmap(dst / "canon.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(pngs), g.CH, g.CW, 3))
    labels = np.zeros((len(pngs), 64), dtype=np.int64)
    for i, png in enumerate(tqdm(pngs, desc=src.name)):
        img, corners, lab = load_render(png)
        canon[i] = g.warp(img, g.homography(corners))
        labels[i] = lab
    canon.flush()
    np.save(dst / "labels.npy", labels)
    (dst / "names.txt").write_text("\n".join(p.stem for p in pngs) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="carpeta con train/ val/ test/ extraídos")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    for split in a.splits:
        convert_split(a.raw / split, a.out / split)


if __name__ == "__main__":
    main()
