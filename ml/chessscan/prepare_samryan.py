"""Convierte samryan18/chess-dataset (MIT, 500 fotos reales con FEN) al formato del contrato.

El dataset no trae esquinas, pero sí una versión rectificada de cada foto
(`labeled_preprocessed/`, recortada justo al área de juego). Las esquinas se
obtienen emparejando esa versión con la original (SIFT + homografía RANSAC) y
proyectando sus bordes; después se refinan con chessscan.corners y se descarta la
foto si el patrón de damero no calza. El FEN está en el nombre del archivo
('-' en vez de '/'); las blancas están siempre del lado de abajo de la foto.

  python -m chessscan.prepare_samryan --raw data/raw/samryan18 --out data/prepared_samryan
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g
from .corners import refine


def checker_corr(canon: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Correlación entre el brillo central de cada casilla y el patrón de damero (±1)."""
    gray = cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = np.array([gray[g.MY + g.S * r + 8:g.MY + g.S * r + 24, g.MX + g.S * c + 8:g.MX + g.S * c + 24].mean()
                  for r in range(8) for c in range(8)])
    chk = np.array([(r + c) % 2 for r in range(8) for c in range(8)]) * 2 - 1
    if mask is not None:
        m, chk = m[mask], chk[mask]
    return float(np.corrcoef(m, chk)[0, 1]) if len(m) > 2 else 0.0


def find_corners(orig: np.ndarray, pre: np.ndarray, sift) -> np.ndarray | None:
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), None)
    if d1 is None or d2 is None:
        return None
    good = [m[0] for m in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 30:
        return None
    H, inl = cv2.findHomography(np.float32([k1[m.queryIdx].pt for m in good]),
                                np.float32([k2[m.trainIdx].pt for m in good]), cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 30:
        return None
    h, w = pre.shape[:2]
    return cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H).reshape(4, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--test-frac", type=float, default=0.2)
    a = ap.parse_args()
    sift = cv2.SIFT_create(nfeatures=4000)
    data = {"train": ([], [], []), "test": ([], [], [])}
    rejected, corners_log = [], {}
    for p in tqdm(sorted((a.raw / "labeled_originals").glob("*.JPG"))):
        pre_p = a.raw / "labeled_preprocessed" / (p.stem + ".png")
        if not pre_p.exists():
            rejected.append((p.name, "sin preprocesada"))
            continue
        orig = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        s = a.max_side / max(orig.shape[:2])
        orig = cv2.resize(orig, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        pre = cv2.cvtColor(cv2.imread(str(pre_p)), cv2.COLOR_BGR2RGB)
        pre = cv2.resize(pre, (800, 800), interpolation=cv2.INTER_AREA)
        c = find_corners(orig, pre, sift)
        if c is None:
            rejected.append((p.name, "sin homografía"))
            continue
        c = refine(orig, g.sort_corners(c), radius=6, iters=5)
        canon = g.warp(orig, g.homography(c))
        corr = checker_corr(canon)
        if abs(corr) < 0.35:
            rejected.append((p.name, f"damero no calza ({corr:+.2f})"))
            continue
        fen = p.stem.split("_")[0].replace("-", "/")  # "_2": otra foto de la misma posición
        split = "test" if int(hashlib.md5(p.stem.split("_")[0].encode()).hexdigest(), 16) % 1000 < a.test_frac * 1000 else "train"
        data[split][0].append(canon)
        data[split][1].append(g.board_labels(fen, "BL"))
        data[split][2].append(p.stem)
        corners_log[p.name] = {"corners": (np.asarray(c) / s).round(1).tolist(), "checker_corr": round(corr, 3)}
    for split, (cs, ls, ns) in data.items():
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "canon.npy", np.stack(cs))
        np.save(d / "labels.npy", np.stack(ls))
        (d / "names.txt").write_text("\n".join(ns) + "\n")
        print(split, len(ns))
    (a.out / "corners.json").write_text(json.dumps(corners_log, indent=0))
    (a.out / "rejected.json").write_text(json.dumps(rejected, indent=1))
    print("descartadas", len(rejected))


if __name__ == "__main__":
    main()
