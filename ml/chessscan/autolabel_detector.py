"""Autoetiquetado de un video con posición fija usando el detector de esquinas entrenado (sin ancla a mano).

Por cuadro: esquinas con `corner_detector.tflite` (dos pasadas), refinadas contra la retícula
(`corners.refine`; se queda con la versión de mejor correlación de damero), canónica y orientación:
de las 4 posiciones posibles de a1 se toma la que más coincide con el clasificador, y el cuadro se
descarta si no le saca ≥ --margin casillas a la segunda. Casillas fuera de la imagen → −1.
Mismo formato de salida que `autolabel_video` (canon.npy, labels.npy, names.txt, frames.json).

  python -m chessscan.autolabel_detector --frames data/raw/user_video/wa_215459.npy \
      --fen "8/8/R1K2k2/4b1n1/3q4/r7/2b1nP1P/8" --out data/real/video7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g
from .corner_net import predict_zoom
from .corners import refine
from .prepare_samryan import checker_corr
from .recover_chessred import classifier, detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True, help="npy [N,H,W,3] RGB")
    ap.add_argument("--fen", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", type=Path, default=Path("export/corner_detector.tflite"))
    ap.add_argument("--classifier", type=Path, default=Path("export/chess_square_classifier_large.tflite"))
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--min-squares", type=int, default=40)
    ap.add_argument("--min-sharpness", type=float, default=30.0)
    ap.add_argument("--min-corr", type=float, default=0.4)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    frames = np.load(a.frames, mmap_mode="r")
    det, cls = detector(a.detector, 8), classifier(a.classifier, 8)
    h, w = frames.shape[1:3]
    ones = np.full((h, w), 255, np.uint8)
    canons, labs, names, log = [], [], [], {}
    stats = {"cuadros": 0, "sin_esquinas": 0, "incompletos": 0, "movidos": 0, "desalineados": 0, "ambiguos": 0}
    for i in tqdm(range(0, len(frames), a.every)):
        stats["cuadros"] += 1
        img = np.asarray(frames[i])
        pts, _ = predict_zoom(det, img)
        if len(pts) < 4:
            stats["sin_esquinas"] += 1
            continue
        best = None
        for c in (g.sort_corners(pts), g.sort_corners(refine(img, g.sort_corners(pts), radius=6.0))):
            H = g.homography(c)
            canon = g.warp(img, H)
            valid = cv2.warpPerspective(ones, H, (g.CW, g.CH), flags=cv2.INTER_NEAREST, borderValue=0)
            mask = np.array([valid[g.MY + g.S * r:g.MY + g.S * (r + 1), g.MX + g.S * q:g.MX + g.S * (q + 1)].min() == 255
                             for r in range(8) for q in range(8)])
            corr = abs(checker_corr(canon, mask))
            if best is None or corr > best[0]:
                best = (corr, c, canon, mask)
        corr, c, canon, mask = best
        sharp = cv2.Laplacian(cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        pred = cls(canon)
        agree = {k: int(((pred == g.board_labels(a.fen, k)) & mask).sum()) for k in ("TL", "TR", "BR", "BL")}
        a1 = max(agree, key=agree.get)
        top, second = sorted(agree.values(), reverse=True)[:2]
        log[str(i)] = {"corners": np.round(c, 1).tolist(), "visible": int(mask.sum()), "sharp": round(float(sharp), 1),
                       "corr": round(float(corr), 3), "a1": a1, "acierto": agree}
        if mask.sum() < a.min_squares:
            stats["incompletos"] += 1
        elif sharp < a.min_sharpness:
            stats["movidos"] += 1
        elif corr < a.min_corr:
            stats["desalineados"] += 1
        elif top - second < a.margin:
            stats["ambiguos"] += 1
        else:
            canons.append(canon)
            labs.append(np.where(mask, g.board_labels(a.fen, a1), -1))
            names.append(f"f{i:05d}")
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "canon.npy", np.stack(canons) if canons else np.zeros((0, g.CH, g.CW, 3), np.uint8))
    np.save(a.out / "labels.npy", np.stack(labs) if labs else np.zeros((0, 64), np.int64))
    (a.out / "names.txt").write_text("\n".join(names) + "\n")
    a1_anchor = log[str(int(names[0][1:]))]["a1"] if names else "BL"  # para eval_video: retícula del primer cuadro guardado
    (a.out / "frames.json").write_text(json.dumps({"fen": a.fen, "a1_anchor": a1_anchor, "stats": stats, "frames": log}, indent=0))
    print(f"{len(names)} cuadros guardados; {stats}", flush=True)


if __name__ == "__main__":
    main()
