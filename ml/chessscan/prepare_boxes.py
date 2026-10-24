"""Convierte datasets de fotos con cajas de piezas (sin FEN ni esquinas) al formato del contrato.

Por foto: esquinas con `corner_detector.tflite` (dos pasadas), homografía y, para cada caja, el punto
de apoyo (abajo al centro, subido 0,35 × el lado menor de la caja: en vista cenital queda cerca del
centro, en vista lateral cerca de la base) proyectado a la canónica → casilla. Las cajas que caen
fuera del tablero (piezas capturadas) se ignoran; las casillas sin caja son vacías.
Se acepta solo si (1) hay 4 esquinas, (2) ninguna casilla recibe dos cajas y (3) la ocupación
(vacía / ocupada) que predice el clasificador v6l coincide en ≥ --min-occ casillas de 64.

  python -m chessscan.prepare_boxes --src dopaul --out data/prepared_dopaul
Fuentes: dopaul (HF dopaul/chess-pieces-merged), acapitani (HF acapitani/chesspiece-detection-yolo),
francesco (HF Francesco/chess-pieces-mjzgj, el «Chess Pieces» público de Roboflow).
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g
from .corner_net import predict_zoom
from .recover_chessred import classifier, detector

# Nombre de clase del dataset → letra FEN.
NAMES = {"black-bishop": "b", "black-king": "k", "black-knight": "n", "black-pawn": "p", "black-queen": "q", "black-rook": "r",
         "white-bishop": "B", "white-king": "K", "white-knight": "N", "white-pawn": "P", "white-queen": "Q", "white-rook": "R",
         "white pawn": "P", "white rook": "R", "white knight": "N", "white bishop": "B", "white queen": "Q", "white king": "K",
         "black pawn": "p", "black rook": "r", "black knight": "n", "black bishop": "b", "black queen": "q", "black king": "k"}


def dopaul(root: Path):
    import pyarrow.parquet as pq
    import yaml
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    for split, f in (("train", "train"), ("train", "valid"), ("test", "test")):
        for r in pq.read_table(root / f"{f}-00000-of-00001.parquet").to_pylist():
            img = cv2.cvtColor(cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), 1), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            a = r["annotations"]
            boxes = [(NAMES[names[c]], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h)
                     for c, x, y, bw, bh in zip(a["class_id"], a["x_center"], a["y_center"], a["width"], a["height"])]
            yield split, f"{r['source_dataset']}/{r['image_id'][:60]}", img, boxes


def yolo_dir(root: Path):
    import yaml
    cfg = next(root.rglob("*.yaml"))
    names = yaml.safe_load(cfg.read_text())["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else names
    if names[0] == "0":  # acapitani: los nombres están solo en los comentarios del yaml
        names = ["white pawn", "white rook", "white knight", "white bishop", "white queen", "white king",
                 "black pawn", "black rook", "black knight", "black bishop", "black queen", "black king", "empty square"]
    for lab in sorted(root.rglob("labels/*/*.txt")):
        split = "test" if lab.parent.name in ("val", "valid", "test") else "train"
        imgs = [p for p in (lab.parent.parent.parent / "images" / lab.parent.name).glob(lab.stem + ".*")]
        if not imgs:
            continue
        img = cv2.imread(str(imgs[0]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = []
        for line in lab.read_text().split("\n"):
            t = line.split()
            if len(t) >= 5 and names[int(t[0])] in NAMES:
                x, y, bw, bh = map(float, t[1:5])
                boxes.append((NAMES[names[int(t[0])]], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
        yield split, lab.stem[:60], img, boxes


FRANCESCO = ["chess-pieces", "bishop", "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook",
             "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook"]


def francesco(root: Path):
    import pyarrow.parquet as pq
    for f in sorted(root.glob("*.parquet")):
        split = "test" if f.name.startswith("test") else "train"
        for r in pq.read_table(f).to_pylist():
            o = r["objects"]
            if any(FRANCESCO[c] not in NAMES for c in o["category"]):
                continue  # «bishop» sin color: no se puede etiquetar
            img = cv2.cvtColor(cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), 1), cv2.COLOR_BGR2RGB)
            boxes = [(NAMES[FRANCESCO[c]], x, y, x + w, y + h) for c, (x, y, w, h) in zip(o["category"], o["bbox"])]
            yield split, f"{f.stem[:5]}_{r['image_id']}", img, boxes


SOURCES = {"francesco": (francesco, Path("data/raw/francesco")), "dopaul": (dopaul, Path("data/raw/dopaul_merged")), "acapitani": (yolo_dir, Path("data/raw/acapitani/x"))}


def assign(boxes, H):
    """Cajas → etiquetas [64] o None si dos cajas caen en la misma casilla."""
    lab = np.zeros(64, np.int64)
    used = set()
    for c, x0, y0, x1, y1 in boxes:
        up = 0.35 * min(x1 - x0, y1 - y0)
        q = cv2.perspectiveTransform(np.float32([[[(x0 + x1) / 2, y1 - up]]]), H)[0, 0]
        col, row = (q[0] - g.MX) / g.S, (q[1] - g.MY) / g.S
        if not (-0.3 < col < 8.3 and -0.3 < row < 8.3):
            continue  # fuera del tablero (capturada)
        i = int(np.clip(row, 0, 7.999)) * 8 + int(np.clip(col, 0, 7.999))
        if i in used:
            return None
        used.add(i)
        lab[i] = g.CLASS_TO_INDEX[c]
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=list(SOURCES), required=True)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--detector", type=Path, default=Path("export/corner_detector.tflite"))
    ap.add_argument("--classifier", type=Path, default=Path("export/chess_square_classifier_large.tflite"))
    ap.add_argument("--min-occ", type=int, default=60)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    fn, root = SOURCES[a.src]
    det, cls = detector(a.detector, a.threads), classifier(a.classifier, a.threads)
    store: dict[str, tuple[list, list, list]] = {}
    stats: dict[str, int] = {}
    log = open(a.out.with_suffix(".log.jsonl"), "w") if a.out.parent.exists() else None
    for split, name, img, boxes in tqdm(fn(a.root or root), desc=a.src):
        s = min(1.0, a.max_side / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            boxes = [(c, x0 * s, y0 * s, x1 * s, y1 * s) for c, x0, y0, x1, y1 in boxes]
        pts, _ = predict_zoom(det, img)
        why = "ok"
        if len(pts) < 4:
            why = "menos de 4 esquinas"
        else:
            H = g.homography(g.sort_corners(pts))
            lab = assign(boxes, H)
            if lab is None:
                why = "dos cajas en una casilla"
            elif (lab > 0).sum() < 2:
                why = "menos de 2 piezas en el tablero"
            else:
                cv = g.warp(img, H)
                occ = int(((cls(cv) > 0) == (lab > 0)).sum())
                if occ < a.min_occ:
                    why = f"ocupación no coincide ({occ}/64)"
                else:
                    c, l, n = store.setdefault(split, ([], [], []))
                    c.append(cv)
                    l.append(lab)
                    n.append(name)
        key = why.split(" (")[0]
        stats[key] = stats.get(key, 0) + 1
        if log:
            log.write(json.dumps({"img": name, "split": split, "motivo": why}) + "\n")
    for split, (c, l, n) in store.items():
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "canon.npy", np.stack(c))
        np.save(d / "labels.npy", np.stack(l))
        (d / "names.txt").write_text("\n".join(n) + "\n")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stats.json").write_text(json.dumps({**stats, **{f"n_{k}": len(v[0]) for k, v in store.items()}}, indent=1))
    print(a.src, stats, {k: len(v[0]) for k, v in store.items()}, flush=True)


if __name__ == "__main__":
    main()
