"""Re-rectifica un video autoetiquetado a otra resolución de cuadro y evalúa.

Lee el video original en orden, reduce cada cuadro etiquetado para que su lado
corto mida `--short` px (INTER_AREA, como haría la app al analizar un cuadro
chico), escala las esquinas guardadas (que están en la escala de `--label-width`)
y escribe un set preparado nuevo que se puede evaluar con `eval_video`.

  python -m chessscan.eval_resolution VID.mp4 --data data/real/video4 --short 480 --out data/real/video4_s480
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--short", type=int, required=True)
    ap.add_argument("--label-width", type=int, default=720)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = json.loads((a.data / "frames.json").read_text())
    names = (a.data / "names.txt").read_text().split()
    want = {int(n[1:]): k for k, n in enumerate(names)}
    labels = np.load(a.data / "labels.npy")
    canon = np.zeros((len(names), g.CH, g.CW, 3), np.uint8)
    cap = cv2.VideoCapture(str(a.video))
    i = 0
    with tqdm(total=len(want)) as bar:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                h, w = f.shape[:2]
                s = a.short / min(h, w)
                small = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                c = np.array(meta["frames"][str(i)]["corners"]) * (small.shape[1] / a.label_width)
                canon[want[i]] = g.warp(rgb, g.homography(c))
                bar.update(1)
            i += 1
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "canon.npy", canon)
    np.save(a.out / "labels.npy", labels)
    (a.out / "names.txt").write_text("\n".join(names) + "\n")
    shutil.copy(a.data / "frames.json", a.out / "frames.json")
    print(f"{len(want)} cuadros a lado corto {a.short} px (ej. {small.shape[1]}x{small.shape[0]})")


if __name__ == "__main__":
    main()
