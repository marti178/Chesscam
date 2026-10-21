"""Prepara fotos reales con FEN conocido y esquinas anotadas al formato canon/labels.

Las esquinas de las fotos reales de chesscog (sin anotar en el original) se
marcaron a ojo y se refinaron con chessscan.corners.refine; quedan en
ml/eval_annotations/chesscog_real_corners.json ({"split/archivo.png": [TL,TR,BR,BL]}).

Uso:  python -m chessscan.prepare_real --images data/raw/chesscog_real \
          --corners eval_annotations/chesscog_real_corners.json --out data/prepared_real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--corners", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ann = json.loads(a.corners.read_text())
    groups: dict[str, list[str]] = {}
    for name in sorted(ann):
        groups.setdefault(name.split("/")[0], []).append(name)
    for split, names in groups.items():
        canon, labels = [], []
        for name in names:
            meta = json.loads((a.images / name).with_suffix(".json").read_text())
            img = cv2.cvtColor(cv2.imread(str(a.images / name)), cv2.COLOR_BGR2RGB)
            canon.append(g.warp(img, g.homography(np.array(ann[name]))))
            labels.append(g.board_labels(meta["fen"], "BL" if meta["white_turn"] else "TR"))
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "canon.npy", np.stack(canon))
        np.save(d / "labels.npy", np.stack(labels))
        (d / "names.txt").write_text("\n".join(names) + "\n")
        print(split, len(names))


if __name__ == "__main__":
    main()
