"""Convierte ChessReD (subconjunto chessred2k, que trae esquinas) al formato del contrato.

ChessReD: Masouris y van Gemert, 4TU.ResearchData, CC BY-NC-SA 4.0 (ver LICENSES.md).
Las esquinas de `annotations.json` se nombran según el tablero, no según la imagen:
top_left = esquina de a8, top_right = h8, bottom_right = h1, bottom_left = a1.
Acá se reordenan a TL, TR, BR, BL de la imagen (CONTRACT §1) y se deduce en qué
esquina de la retícula quedó a1. Como control, el centro de la caja de cada pieza
proyectado con la homografía tiene que caer cerca de su casilla anotada.

Acepta imágenes originales (3072 px) o las preprocesadas del repo de los autores:
las esquinas se escalan al tamaño real de cada archivo.

  python -m chessscan.prepare_chessred --ann annotations.json --images data/raw/chessred/images \
      --out data/prepared_chessred
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g

CAT = {0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K", 6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k"}
BOARD_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]  # a8, h8, h1, a1
A1_AT = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}


def find_image(root: Path, rel: str, name: str) -> Path | None:
    for p in (root / rel, root / Path(rel).relative_to("images") if rel.startswith("images") else root / rel,
              root / name):
        if p.exists():
            return p
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1280)
    a = ap.parse_args()
    ann = json.loads(a.ann.read_text())
    images = {im["id"]: im for im in ann["images"]}
    corners = {c["image_id"]: c["corners"] for c in ann["annotations"]["corners"]}
    pieces: dict[int, list] = {}
    for p in ann["annotations"]["pieces"]:
        pieces.setdefault(p["image_id"], []).append(p)
    index = {}
    for split in ("train", "val", "test"):
        ids = ann["splits"]["chessred2k"][split]["image_ids"]
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        canons, labels, names, rejected = [], [], [], []
        for iid in tqdm(ids, desc=split):
            im = images[iid]
            path = find_image(a.images, im["path"], im["file_name"])
            if path is None:
                rejected.append((im["file_name"], "no está la imagen"))
                continue
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            sx, sy = img.shape[1] / im["width"], img.shape[0] / im["height"]
            s = min(1.0, a.max_side / max(img.shape[:2]))
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            k = np.array([sx * s, sy * s])
            board = np.array([corners[iid][n] for n in BOARD_ORDER], np.float64) * k
            # Orden de imagen: rotar la lista del tablero hasta que el primero sea el TL de la imagen.
            tl = g.sort_corners(board)[0]
            start = int(np.argmin(np.linalg.norm(board - tl, axis=1)))
            img_c = np.roll(board, -start, axis=0)
            # El índice (en orden de imagen) donde quedó a1 (índice 3 en BOARD_ORDER).
            a1 = A1_AT[(3 - start) % 4]
            H = g.homography(img_c)
            lab = np.zeros(64, np.int64)
            sq_of = {g.grid_to_square(r, c, a1): r * 8 + c for r in range(8) for c in range(8)}
            bad = 0
            for p in pieces.get(iid, []):
                if p["category_id"] == 12:
                    continue
                i = sq_of[p["chessboard_position"]]
                lab[i] = g.CLASS_TO_INDEX[CAT[p["category_id"]]]
                # Control: la base de la caja (abajo al centro) cae en la casilla o justo debajo.
                x, y, w, h = p["bbox"]
                q = cv2.perspectiveTransform(np.float32([[[(x + w / 2) * k[0], (y + h) * k[1]]]]), H)[0, 0]
                r_est, c_est = (q[1] - g.MY) / g.S, (q[0] - g.MX) / g.S
                if abs(c_est - (i % 8 + 0.5)) > 1.2 or not (-0.5 < r_est - i // 8 < 2.2):
                    bad += 1
            n_p = sum(1 for p in pieces.get(iid, []) if p["category_id"] != 12)
            if n_p and bad > max(2, 0.2 * n_p):
                rejected.append((im["file_name"], f"{bad}/{n_p} cajas fuera de su casilla"))
                continue
            canons.append(g.warp(img, H))
            labels.append(lab)
            names.append(f"{im['file_name']} a1={a1}")
        np.save(d / "canon.npy", np.stack(canons))
        np.save(d / "labels.npy", np.stack(labels))
        (d / "names.txt").write_text("\n".join(names) + "\n")
        (d / "rejected.json").write_text(json.dumps(rejected, indent=1))
        index[split] = {"ok": len(names), "rejected": len(rejected)}
        print(split, index[split])


if __name__ == "__main__":
    main()
