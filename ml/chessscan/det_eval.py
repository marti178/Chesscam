"""Evaluación del detector de piezas con las mismas métricas que el clasificador (casillas y piezas por
cuadro, fusión temporal y recall por clase), sobre las fotos ORIGINALES con las esquinas conocidas.

Decodificación y asignación (CONTRACT_DETECTOR.md): recorte del tablero (`det_data.crop_box`, lado
mayor ≤ 640), detección con NMS agnóstica, punto de apoyo de cada caja (abajo al centro, subido
0,35 × el lado menor) → homografía → casilla; cada casilla se queda con la detección de mayor
confianza. Para fusionar se arma una probabilidad por casilla: con detección de confianza p,
q = sigmoid(logit(p) − logit(t)) para su clase y el resto para «vacía»; sin detección, «vacía» 0,95.

Corre en el entorno de Ultralytics (.venv-yolo):
  python -m chessscan.det_eval --model runs/det_v1/weights/best.pt --out runs/det_v1/eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g

MAX = 640
RAW = Path("data/raw/user_video")
VIDEOS = {  # nombre: (carpeta autoetiquetada, cuadros originales, tramo)
    "132356": ("data/real/video4", RAW / "VID_20261022_132356_720.npy", None),
    "133425": ("data/real/video5", RAW / "VID_20261022_133425_720.npy", None),
    "tablero_nuevo_215505": ("data/real/video8", RAW / "wa_215505.npy", None),
    "libro_monitor": ("data/real/video6book", RAW / "VID_20261022_134155_720.npy", (1000, 1412)),
    "2d_real": ("data/real/video2", RAW / "frames2_720.npy", None),
}
STATIC = {  # nombre: (carpeta preparada, carpeta de imágenes, json de esquinas)
    "chesscog_real": ("data/prepared_real/test", Path("data/raw/chesscog_real"), Path("eval_annotations/chesscog_real_corners.json")),
    "alekhine": ("data/prepared_book_real/test", Path("C:/Users/MartinM/ml_fotos/libros"), Path("eval_annotations/libro_alekhine_01_corners.json")),
}


def crop_box(corners, w, h):
    lo, hi = corners.min(0), corners.max(0)
    bw, bh = hi - lo
    return (int(max(0, lo[0] - 0.08 * bw)), int(max(0, lo[1] - 0.35 * bh)),
            int(min(w, hi[0] + 0.08 * bw)), int(min(h, hi[1] + 0.05 * bh)))


class Detector:
    def __init__(self, model: Path, conf: float, iou: float):
        from ultralytics import YOLO
        self.m, self.conf, self.iou = YOLO(str(model)), conf, iou

    def __call__(self, rgb: np.ndarray, corners: np.ndarray):
        """→ [(clase 1..12, p, x0, y0, x1, y1)] en píxeles de la imagen original."""
        h, w = rgb.shape[:2]
        X0, Y0, X1, Y1 = crop_box(corners, w, h)
        crop = rgb[Y0:Y1, X0:X1]
        s = min(1.0, MAX / max(crop.shape[:2]))
        if s < 1:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r = self.m.predict(crop[..., ::-1], imgsz=MAX, conf=self.conf, iou=self.iou, agnostic_nms=True, verbose=False)[0]
        out = []
        for (x0, y0, x1, y1), c, p in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy()):
            out.append((int(c) + 1, float(p), x0 / s + X0, y0 / s + Y0, x1 / s + X0, y1 / s + Y0))
        return out


def assign(dets, corners, t: float) -> np.ndarray:
    """Detecciones → log-probabilidades [64, 13] en la retícula de `corners` (TL, TR, BR, BL)."""
    H = g.homography(corners)
    best = {}
    for c, p, x0, y0, x1, y1 in dets:
        up = 0.35 * min(x1 - x0, y1 - y0)
        q = cv2.perspectiveTransform(np.float32([[[(x0 + x1) / 2, y1 - up]]]), H)[0, 0]
        col, row = (q[0] - g.MX) / g.S, (q[1] - g.MY) / g.S
        if not (0 <= col < 8 and 0 <= row < 8):
            continue
        i = int(row) * 8 + int(col)
        if i not in best or p > best[i][1]:
            best[i] = (c, p)
    P = np.full((64, 13), 0.05 / 12)
    P[:, 0] = 0.95
    lt = np.log(t / (1 - t))
    for i, (c, p) in best.items():
        p = min(max(p, 1e-4), 1 - 1e-4)
        qq = 1 / (1 + np.exp(-(np.log(p / (1 - p)) - lt)))
        P[i] = (1 - qq) * 0.05 / 11
        P[i, 0] = (1 - qq) * 0.95
        P[i, c] = qq
    return np.log(P)


def rotation_to_ref(lab, ref):
    L, R = lab.reshape(8, 8), ref.reshape(8, 8)
    vis = L >= 0
    return max(range(4), key=lambda k: ((np.rot90(R, k) == L) & vis).sum())


def summarize(logp, labels, ref, windows=(1, 60, 0)):
    """Igual que eval_video: fusión por casilla física sumando log-probabilidades."""
    n = len(labels)
    phys_lp = np.zeros_like(logp)
    phys_vis = np.zeros((n, 64), bool)
    for j in range(n):
        k = rotation_to_ref(labels[j], ref)
        inv = np.rot90(np.arange(64).reshape(8, 8), k).reshape(64)
        phys_lp[j, inv] = logp[j]
        phys_vis[j, inv] = labels[j] >= 0
    res = {"n_frames": n}
    for W in windows:
        W_eff = n if W == 0 else min(W, n)
        ok = tot = occ_ok = occ_tot = 0
        for s in range(0, n - W_eff + 1, max(1, W_eff // 2) if W_eff > 1 else 1):
            acc = (phys_lp[s:s + W_eff] * phys_vis[s:s + W_eff, :, None]).sum(0)
            seen = phys_vis[s:s + W_eff].any(0)
            pred = acc.argmax(-1)
            ok += (pred[seen] == ref[seen]).sum(); tot += seen.sum()
            m = seen & (ref != 0)
            occ_ok += (pred[m] == ref[m]).sum(); occ_tot += m.sum()
        res["todo_el_tramo" if W == 0 else f"ventana_{W}"] = {"square_acc": ok / tot, "occupied_acc": occ_ok / max(occ_tot, 1)}
    pred = logp.argmax(-1)
    vis = labels >= 0
    conf = np.zeros((13, 13), int)
    np.add.at(conf, (labels[vis], pred[vis]), 1)
    res["per_class_recall_1frame"] = {c: (float(conf[i, i] / conf[i].sum()) if conf[i].sum() else None) for i, c in enumerate(g.CLASSES)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.1, help="umbral mínimo de la detección cruda")
    ap.add_argument("--t", type=float, default=0.35, help="confianza que equivale a 50 % en la fusión")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--every", type=int, default=1, help="1 de cada N cuadros de video")
    ap.add_argument("--sets", nargs="*", default=list(VIDEOS) + list(STATIC))
    a = ap.parse_args()
    det = Detector(a.model, a.conf, a.iou)
    res = {}
    for name in a.sets:
        if name in VIDEOS:
            d, raw, rng_ = VIDEOS[name]
            d = Path(d)
            meta = json.loads((d / "frames.json").read_text())
            names = (d / "names.txt").read_text().split()
            labels_all = np.load(d / "labels.npy")
            frames = np.load(raw, mmap_mode="r")
            sel = [k for k, n in enumerate(names) if rng_ is None or rng_[0] <= int(n[1:]) <= rng_[1]][::a.every]
            lps, labs = [], []
            for k in sel:
                i = int(names[k][1:])
                c = np.array(meta["frames"][str(i)]["corners"], float)
                lps.append(assign(det(np.asarray(frames[i]), c), c, a.t))
                labs.append(labels_all[k])
            ref = g.board_labels(meta["fen"], meta["a1_anchor"])
            res[name] = summarize(np.stack(lps), np.stack(labs), ref)
        else:
            d, imgs, cj = STATIC[name]
            corners = json.loads(cj.read_text())
            names = (Path(d) / "names.txt").read_text().split()
            labels = np.load(Path(d) / "labels.npy")
            ok = tot = occ_ok = occ_tot = 0
            for n, lab in zip(names, labels):
                key = n if n in corners else n + ".png"
                c = np.array(corners[key], float)
                rgb = cv2.cvtColor(cv2.imread(str(imgs / key)), cv2.COLOR_BGR2RGB)
                pred = assign(det(rgb, c), c, a.t).argmax(-1)
                v = lab >= 0
                ok += (pred[v] == lab[v]).sum(); tot += v.sum()
                m = v & (lab != 0)
                occ_ok += (pred[m] == lab[m]).sum(); occ_tot += m.sum()
            res[name] = {"ventana_1": {"square_acc": ok / tot, "occupied_acc": occ_ok / max(occ_tot, 1)}, "n": len(names)}
        r = res[name]
        print(name, {k: v for k, v in r.items() if k != "per_class_recall_1frame"}, flush=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
