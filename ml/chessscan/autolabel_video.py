"""Autoetiquetado desde un cuadro ancla, siguiendo el tablero hacia adelante y hacia atrás.

Variante de `autolabel` para videos en los que el tablero entra completo en pocos
cuadros, o en los que el primer cuadro no sirve. Se dan las esquinas en un cuadro
cualquiera (pueden quedar fuera de la imagen) y en qué esquina de la retícula
está a1 en ese cuadro. Después:
  - se sigue el tablero cuadro a cuadro con SIFT + homografía (tracker de autolabel),
  - en cada cuadro se refinan las esquinas contra las líneas internas de la
    retícula (radio chico) para que no se acumule deriva,
  - se reordenan las esquinas a TL, TR, BR, BL de la imagen y se rotan las etiquetas.
Se guardan solo los cuadros con las 4 esquinas dentro de la imagen (`--margin`),
nítidos y con el damero bien alineado.

  python -m chessscan.autolabel_video video.mp4 --anchor 1100 \
      --anchor-corners "86.9,326.1 655.3,249.7 933.3,688.8 173.4,848.0" --a1 BL \
      --fen "rr4k1/R1Q2ppp/4pq2/8/7n/3P4/2P2NPP/5R1K" --out data/real/video1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g
from .autolabel import Tracker, upright
from .corners import refine
from .prepare_samryan import checker_corr


def load_frames(video: Path, width: int, cache: Path | None) -> np.ndarray:
    if cache and cache.exists():
        return np.load(cache, mmap_mode="r")
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:  # lectura secuencial: el salto con CAP_PROP_POS_FRAMES no es exacto en H.264
        ok, f = cap.read()
        if not ok:
            break
        h = int(round(f.shape[0] * width / f.shape[1]))
        frames.append(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    arr = np.stack(frames)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    print(f"{len(arr)} cuadros (el contenedor dice {n})")
    return arr


def track(frames, anchor: int, corners0: np.ndarray, order) -> dict[int, np.ndarray]:
    out = {anchor: corners0}
    tr = Tracker(cv2.cvtColor(np.asarray(frames[anchor]), cv2.COLOR_RGB2GRAY), corners0)
    for i in tqdm(order, desc="seguimiento"):
        img = np.asarray(frames[i])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        c, _ = tr.track(gray)
        if c is None:
            break  # se perdió: no seguir más en esta dirección
        c = refine(img, c, radius=4, iters=3, inner_only=True)
        tr._set_key(gray, c)  # el cuadro refinado pasa a ser el clave: sin deriva acumulada
        out[i] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--anchor", type=int, required=True)
    ap.add_argument("--anchor-corners", required=True, help="'x,y x,y x,y x,y' TL TR BR BL en el cuadro ancla (escala --width)")
    ap.add_argument("--a1", required=True, choices=["BL", "TR", "TL", "BR"], help="esquina de a1 en el cuadro ancla")
    ap.add_argument("--fen", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--cache", type=Path, help="npy con los cuadros ya reducidos")
    ap.add_argument("--margin", type=float, default=0.0, help="px que una esquina puede quedar fuera")
    ap.add_argument("--min-sharpness", type=float, default=40.0)
    ap.add_argument("--min-corr", type=float, default=0.4)
    ap.add_argument("--min-squares", type=int, default=0,
                    help="si > 0: aceptar cuadros con al menos N casillas enteras en la imagen (máscara por casilla) "
                         "en vez de exigir las 4 esquinas adentro")
    a = ap.parse_args()

    frames = load_frames(a.video, a.width, a.cache)
    c0 = np.array([[float(v) for v in p.split(",")] for p in a.anchor_corners.split()])
    labels0 = g.board_labels(a.fen, a.a1)
    tracked = track(frames, a.anchor, c0, range(a.anchor + 1, len(frames)))
    tracked.update(track(frames, a.anchor, c0, range(a.anchor - 1, -1, -1)))

    h, w = frames.shape[1:3]
    canons, labs, masks, names, log = [], [], [], [], {}
    ones = np.full((h, w), 255, np.uint8)
    stats = {"cuadros": len(frames), "seguidos": len(tracked), "incompletos": 0, "movidos": 0, "desalineados": 0}
    for i in sorted(tracked):
        c, lab = upright(tracked[i], labels0)
        inside = ((c[:, 0] >= -a.margin) & (c[:, 0] <= w - 1 + a.margin) &
                  (c[:, 1] >= -a.margin) & (c[:, 1] <= h - 1 + a.margin)).all()
        H = g.homography(c)
        canon = g.warp(np.asarray(frames[i]), H)
        valid = cv2.warpPerspective(ones, H, (g.CW, g.CH), flags=cv2.INTER_NEAREST, borderValue=0)
        mask = np.array([valid[g.MY + g.S * r:g.MY + g.S * (r + 1), g.MX + g.S * q:g.MX + g.S * (q + 1)].min() == 255
                         for r in range(8) for q in range(8)])
        sharp = cv2.Laplacian(cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        corr = checker_corr(canon, mask)
        log[i] = {"corners": c.round(1).tolist(), "inside": bool(inside), "visible": int(mask.sum()),
                  "sharp": round(sharp, 1), "corr": round(corr, 3)}
        if (a.min_squares and mask.sum() < a.min_squares) or (not a.min_squares and not inside):
            stats["incompletos"] += 1
            continue
        if sharp < a.min_sharpness:
            stats["movidos"] += 1
            continue
        if abs(corr) < a.min_corr:
            stats["desalineados"] += 1
            continue
        canons.append(canon)
        labs.append(np.where(mask, lab, -1) if a.min_squares else lab)
        masks.append(mask)
        names.append(f"f{i:05d}")
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "canon.npy", np.stack(canons) if canons else np.zeros((0, g.CH, g.CW, 3), np.uint8))
    np.save(a.out / "labels.npy", np.stack(labs) if labs else np.zeros((0, 64), np.int64))
    (a.out / "names.txt").write_text("\n".join(names) + "\n")
    (a.out / "frames.json").write_text(json.dumps({"fen": a.fen, "anchor": a.anchor, "a1_anchor": a.a1,
                                                   "stats": stats, "frames": log}, indent=0))
    print(f"{len(names)} cuadros guardados; {stats}")


if __name__ == "__main__":
    main()
