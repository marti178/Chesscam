"""Dataset YOLO (12 clases) para el detector de piezas, en el recorte del tablero de CONTRACT_DETECTOR.md.

Cada imagen: la foto original recortada al rectángulo de las 4 esquinas, ampliado (`crop_box`) y
reducida a lado mayor ≤ 640. Cajas por fuente:
  ChessReD2K — sus cajas y esquinas; ChessSight — cajas modales exactas (piezas visibles) y esquinas;
  acapitani, dopaul, Roboflow (francesco) — sus cajas y esquinas del detector de esquinas;
  sintético 2D y de libro — la casilla proyectada (piezas planas), esquinas exactas;
  videos de entrenamiento del usuario — cilindros de altura nominal proyectados con la pose de la cámara
  (el libro del monitor, con casillas).
Clases en el orden de CONTRACT.md sin «empty»: P N B R Q K p n b r q k → 0..11.

  python -m chessscan.det_data --out data/det
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g

MAX = 640
YOLO = g.CLASSES[1:]


def crop_box(corners: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    """Rectángulo de recorte: el de las esquinas + 8 % a los costados, 5 % abajo y 35 % arriba (piezas altas)."""
    lo, hi = corners.min(0), corners.max(0)
    bw, bh = hi - lo
    x0, x1 = lo[0] - 0.08 * bw, hi[0] + 0.08 * bw
    y0, y1 = lo[1] - 0.35 * bh, hi[1] + 0.05 * bh
    return int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))


class Writer:
    def __init__(self, out: Path):
        self.out, self.n = out, {}

    def add(self, split, name, rgb, corners, boxes):
        """boxes: [(clase FEN, x0, y0, x1, y1)] en píxeles de `rgb`."""
        h, w = rgb.shape[:2]
        X0, Y0, X1, Y1 = crop_box(np.asarray(corners, float), w, h)
        if X1 - X0 < 32 or Y1 - Y0 < 32:
            return
        crop = rgb[Y0:Y1, X0:X1]
        s = min(1.0, MAX / max(crop.shape[:2]))
        if s < 1:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ch, cw = crop.shape[:2]
        lines = []
        for c, x0, y0, x1, y1 in boxes:
            a0 = max(0, (x1 - x0) * (y1 - y0))
            bx0, by0, bx1, by1 = max(x0, X0), max(y0, Y0), min(x1, X1), min(y1, Y1)
            if a0 <= 0 or bx1 <= bx0 or by1 <= by0 or (bx1 - bx0) * (by1 - by0) < 0.5 * a0:
                continue
            bx0, by0, bx1, by1 = [(v - o) * s for v, o in ((bx0, X0), (by0, Y0), (bx1, X0), (by1, Y0))]
            lines.append(f"{YOLO.index(c)} {(bx0 + bx1) / 2 / cw:.5f} {(by0 + by1) / 2 / ch:.5f} {(bx1 - bx0) / cw:.5f} {(by1 - by0) / ch:.5f}")
        for sub in ("images", "labels"):
            (self.out / sub / split).mkdir(parents=True, exist_ok=True)
        stem = name.replace("/", "_").replace("\\", "_")
        cv2.imwrite(str(self.out / "images" / split / f"{stem}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        (self.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        self.n[split] = self.n.get(split, 0) + 1


def chessred(w: Writer, ann: Path, images: Path):
    from .prepare_chessred import BOARD_ORDER, CAT, find_image
    a = json.loads(ann.read_text())
    im = {i["id"]: i for i in a["images"]}
    cor = {c["image_id"]: c["corners"] for c in a["annotations"]["corners"]}
    pcs: dict[int, list] = {}
    for p in a["annotations"]["pieces"]:
        if p["category_id"] != 12:
            pcs.setdefault(p["image_id"], []).append(p)
    for split in ("train", "val", "test"):
        for iid in tqdm(a["splits"]["chessred2k"][split]["image_ids"], desc=f"chessred {split}"):
            info = im[iid]
            img = cv2.cvtColor(cv2.imread(str(find_image(images, info["path"], info["file_name"]))), cv2.COLOR_BGR2RGB)
            k = img.shape[1] / info["width"]
            c = np.array([cor[iid][n] for n in BOARD_ORDER]) * k
            boxes = [(CAT[p["category_id"]], p["bbox"][0] * k, p["bbox"][1] * k, (p["bbox"][0] + p["bbox"][2]) * k,
                      (p["bbox"][1] + p["bbox"][3]) * k) for p in pcs.get(iid, [])]
            w.add(split, f"chessred_{iid}", img, c, boxes)


def chesssight(w: Writer, root: Path, n_train: int, n_val: int):
    import pyarrow.parquet as pq
    for split, pat, mx in (("train", "train-*.parquet", n_train), ("val", "validation-*.parquet", n_val)):
        n = 0
        for f in sorted(root.glob(pat)):
            for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["image", "id", "corners_px", "pieces"]):
                for r in b.to_pylist():
                    if n >= mx:
                        break
                    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), 1), cv2.COLOR_BGR2RGB)
                    boxes = [(g.CLASSES[p["class_id"]], *p["bbox"]) for p in r["pieces"] if p["visible"]]
                    w.add(split, f"chesssight_{r['id']}", img, np.array(r["corners_px"]), boxes)
                    n += 1
            if n >= mx:
                break


def box_sources(w: Writer):
    from .corner_net import predict_zoom
    from .prepare_boxes import SOURCES
    from .recover_chessred import detector
    det = detector(Path("export/corner_detector.tflite"), 8)
    for src, (fn, root) in SOURCES.items():
        for split, name, img, boxes in tqdm(fn(root), desc=src):
            pts, _ = predict_zoom(det, img)
            if len(pts) == 4:
                w.add("val" if split == "test" else "train", f"{src}_{name}", img, pts, boxes)


def flat_boxes(corners, labels):
    """Piezas planas (2D y libro): la caja es la casilla proyectada."""
    Hinv = np.linalg.inv(g.homography(np.asarray(corners, float)))
    out = []
    for i in np.flatnonzero(labels > 0):
        r, c = divmod(int(i), 8)
        q = np.float32([[[g.MX + c * g.S, g.MY + r * g.S]], [[g.MX + (c + 1) * g.S, g.MY + r * g.S]],
                        [[g.MX + (c + 1) * g.S, g.MY + (r + 1) * g.S]], [[g.MX + c * g.S, g.MY + (r + 1) * g.S]]])
        p = cv2.perspectiveTransform(q, Hinv).reshape(4, 2)
        out.append((g.CLASSES[labels[i]], *p.min(0), *p.max(0)))
    return out


def synthetic(w: Writer, n2d: int, nbook: int):
    from . import synth2d, synthbook
    seen2d = synth2d.load_sets(Path("data/pieces_png"), "train")
    synth2d.load_textures(Path("data/raw/boards"))
    rng = np.random.default_rng([78, 0])
    for i in tqdm(range(n2d), desc="2d"):
        photo, c, lab, *_ = synth2d.sample(seen2d, rng)
        w.add("train", f"synth2d_{i:05d}", photo, c, flat_boxes(c, lab))
    seenb = synthbook.load_book_sets(Path("data/pieces_png"), "train")
    rng = np.random.default_rng([78, 1])
    for i in tqdm(range(nbook), desc="libro"):
        photo, c, lab, *_ = synthbook.sample(seenb, rng)
        w.add("train", f"synthbook_{i:05d}", photo, c, flat_boxes(c, lab))


# Altura nominal (en lados de casilla) de piezas Staunton y radio de la base.
HEIGHT = {"P": 0.85, "R": 0.95, "N": 1.15, "B": 1.3, "Q": 1.55, "K": 1.75}
RADIUS = 0.33


def camera_pose(corners: np.ndarray, w: int, h: int):
    """Pose de la cámara a partir de la homografía tablero → imagen (tablero en unidades de casilla,
    z = 0), con el punto principal en el centro y la focal deducida de la ortogonalidad de r1 y r2."""
    Hb = cv2.getPerspectiveTransform(np.float32([[0, 0], [8, 0], [8, 8], [0, 8]]), np.float32(corners)).astype(np.float64)
    C = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])
    Hc = C @ Hb
    num = -(Hc[0, 0] * Hc[0, 1] + Hc[1, 0] * Hc[1, 1])
    den = Hc[2, 0] * Hc[2, 1]
    f = np.sqrt(num / den) if den != 0 and num / den > 0 else 1.2 * max(w, h)
    f = float(np.clip(f, 0.5 * max(w, h), 4 * max(w, h)))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    M = np.linalg.inv(K) @ Hb
    lam = np.linalg.norm(M[:, 0])
    r1, r2, t = M[:, 0] / lam, M[:, 1] / lam, M[:, 2] / lam
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], 1)
    def proj(P):
        q = (K @ (R @ P.T + t[:, None])).T
        return q[:, :2] / q[:, 2:3]
    # Las piezas «suben» en la imagen: si z+ baja, invertir r3.
    if proj(np.array([[4, 4, 1.0]]))[0, 1] > proj(np.array([[4, 4, 0.0]]))[0, 1]:
        R[:, 2] = -r3
    return proj


def cylinder_boxes(corners, labels, w, h):
    """Cajas aproximadas: cilindro de altura nominal por tipo de pieza, proyectado con la pose."""
    proj = camera_pose(np.asarray(corners, float), w, h)
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    out = []
    for i in np.flatnonzero(labels > 0):
        r, c = divmod(int(i), 8)
        name = g.CLASSES[labels[i]]
        ring = np.stack([c + 0.5 + RADIUS * np.cos(a), r + 0.5 + RADIUS * np.sin(a), np.zeros(16)], 1)
        top = ring.copy(); top[:, 2] = HEIGHT[name.upper()]
        q = proj(np.concatenate([ring, top]))
        out.append((name, *q.min(0), *q.max(0)))
    return out


def user_videos(w: Writer, every: int):
    """Videos de entrenamiento del usuario (nunca los de test): físicos con cilindros, el libro con casillas."""
    specs = [("video1", "data/raw/user_video/frames_720.npy", None, "cil"),
             ("video3", "data/raw/user_video/VID_20261022_132028_720.npy", None, "cil"),
             ("video7", "data/raw/user_video/wa_215459.npy", None, "cil"),
             ("video6book", "data/raw/user_video/VID_20261022_134155_720.npy", (0, 899), "plano")]
    for name, raw, rng_, kind in specs:
        d = Path("data/real") / name
        meta = json.loads((d / "frames.json").read_text())
        names = (d / "names.txt").read_text().split()
        labels = np.load(d / "labels.npy")
        frames = np.load(raw, mmap_mode="r")
        for k in tqdm(range(0, len(names), every), desc=name):
            i = int(names[k][1:])
            if rng_ and not rng_[0] <= i <= rng_[1]:
                continue
            c = np.array(meta["frames"][str(i)]["corners"], float)
            img = np.asarray(frames[i])
            lab = np.maximum(labels[k], 0)  # las casillas fuera de la imagen no generan caja visible igual
            boxes = cylinder_boxes(c, lab, img.shape[1], img.shape[0]) if kind == "cil" else flat_boxes(c, lab)
            w.add("train", f"{name}_{i:05d}", img, c, boxes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/det"))
    ap.add_argument("--chesssight", type=int, nargs=2, default=[3000, 300])
    ap.add_argument("--n2d", type=int, default=800)
    ap.add_argument("--nbook", type=int, default=800)
    ap.add_argument("--video-every", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=["chessred", "chesssight", "boxes", "synthetic", "videos"])
    a = ap.parse_args()
    w = Writer(a.out)
    if "chessred" in a.only:
        chessred(w, Path("C:/Users/MartinM/ml_fotos/chessred/annotations.json"), Path("data/raw/chessred"))
    if "chesssight" in a.only:
        chesssight(w, Path("data/raw/chesssight"), *a.chesssight)
    if "boxes" in a.only:
        box_sources(w)
    if "synthetic" in a.only:
        synthetic(w, a.n2d, a.nbook)
    if "videos" in a.only:
        user_videos(w, a.video_every)
    (a.out / "data.yaml").write_text(f"path: {a.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                                     f"names: {json.dumps(dict(enumerate(YOLO)))}\n")
    print(w.n, flush=True)


if __name__ == "__main__":
    main()
