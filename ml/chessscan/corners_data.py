"""Arma el dataset del detector de esquinas: imágenes (lado mayor 512) + 4 esquinas del área de juego.

Fuentes (todas con esquinas exactas o verificadas; las licencias están en LICENSES.md):
  chesscog sintético (renders + JSON), ChessReD chessred2k, surawut/chessboard-dataset-yolo,
  samryan18 (esquinas calculadas), videos del usuario (esquinas del autoetiquetado),
  sintético 2D y de libros (esquinas exactas del generador) y las 27 fotos reales de chesscog.
Split: test = chessred2k test, videos 132356/133425, libro del monitor 1000–1412, chesscog real,
samryan test; val = chessred2k val, surawut valid, chesscog val; train = el resto.

  python -m chessscan.corners_data --out data/corners
Salida: <out>/<split>/<fuente>_<id>.jpg y <out>/<split>.jsonl ({"img", "corners": [[x,y]×4], "src"}).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

MAX = 512


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.files = {}

    def add(self, split: str, src: str, key: str, rgb: np.ndarray, corners: np.ndarray) -> None:
        s = MAX / max(rgb.shape[:2])
        if s < 1:
            rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            corners = corners * s
        d = self.out / split
        d.mkdir(parents=True, exist_ok=True)
        name = f"{src}_{key}.jpg"
        cv2.imwrite(str(d / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if split not in self.files:
            self.files[split] = open(self.out / f"{split}.jsonl", "w")
        self.files[split].write(json.dumps({"img": f"{split}/{name}", "corners": np.round(corners, 2).tolist(), "src": src}) + "\n")

    def close(self):
        for f in self.files.values():
            f.close()


def rgb_of(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def chesscog_synth(w: Writer, root: Path, every: int):
    for split, sub in (("train", "train"), ("val", "val")):
        for p in tqdm(sorted((root / sub).glob("*.png"))[::every], desc=f"chesscog {split}"):
            meta = json.loads(p.with_suffix(".json").read_text())
            w.add(split, "chesscog", p.stem, rgb_of(p), np.array(meta["corners"], float))


def chessred(w: Writer, ann: Path, images: Path):
    a = json.loads(ann.read_text())
    im = {i["id"]: i for i in a["images"]}
    cor = {c["image_id"]: c["corners"] for c in a["annotations"]["corners"]}
    for split in ("train", "val", "test"):
        for iid in tqdm(a["splits"]["chessred2k"][split]["image_ids"], desc=f"chessred {split}"):
            info = im[iid]
            img = rgb_of(images / info["path"])
            k = img.shape[1] / info["width"]
            c = np.array([cor[iid][n] for n in ("top_left", "top_right", "bottom_right", "bottom_left")]) * k
            w.add(split, "chessred", str(iid), img, c)


def surawut(w: Writer, root: Path):
    for split, sub in (("train", "train"), ("val", "valid")):
        for p in tqdm(sorted((root / sub / "images").glob("*.jpg")), desc=f"surawut {split}"):
            lab = (root / sub / "labels" / (p.stem + ".txt")).read_text().split()
            if len(lab) < 17:
                continue
            img = rgb_of(p)
            h, wd = img.shape[:2]
            kp = np.array([float(t) for t in lab[5:17]]).reshape(4, 3)[:, :2] * [wd, h]
            w.add(split, "surawut", p.stem[:60], img, kp)


def samryan(w: Writer, raw: Path, prepared: Path):
    log = json.loads((prepared / "corners.json").read_text())
    test = set((prepared / "test" / "names.txt").read_text().split())
    for name, v in tqdm(log.items(), desc="samryan"):
        img = rgb_of(raw / "labeled_originals" / name)
        split = "test" if Path(name).stem in test else "train"
        w.add(split, "samryan", Path(name).stem[:60], img, np.array(v["corners"]))


def user_videos(w: Writer, every: int):
    specs = [("video1", "VID_20261022_104414", "train", None), ("video3", "VID_20261022_132028", "train", None),
             ("video2", "VID_20261022_105447", "train", None), ("video6book", "VID_20261022_134155", "train", (0, 899)),
             ("video6book", "VID_20261022_134155", "test", (1000, 1412)),
             ("video4", "VID_20261022_132356", "test", None), ("video5", "VID_20261022_133425", "test", None)]
    caches = {"VID_20261022_104414": "frames_720.npy", "VID_20261022_105447": "frames2_720.npy"}
    for name, vid, split, rng_ in specs:
        meta = json.loads(Path(f"data/real/{name}/frames.json").read_text())
        frames = np.load(Path("data/raw/user_video") / caches.get(vid, f"{vid}_720.npy"), mmap_mode="r")
        keys = sorted(int(k) for k in meta["frames"])
        keys = [k for k in keys if rng_ is None or rng_[0] <= k <= rng_[1]][::every]
        for i in tqdm(keys, desc=f"{name} {split}"):
            w.add(split, name, f"{i:05d}", np.asarray(frames[i]), np.array(meta["frames"][str(i)]["corners"]))


def synthetic(w: Writer, n2d: int, nbook: int):
    from . import synth2d, synthbook
    from .pieces2d import TEST_SETS  # noqa: F401
    seen2d = synth2d.load_sets(Path("data/pieces_png"), "train")
    synth2d.load_textures(Path("data/raw/boards"))
    rng = np.random.default_rng([77, 0])
    for i in tqdm(range(n2d), desc="2d"):
        photo, c, *_ = synth2d.sample(seen2d, rng)
        w.add("train", "synth2d", f"{i:05d}", photo, c)
    seenb = synthbook.load_book_sets(Path("data/pieces_png"), "train")
    rng = np.random.default_rng([77, 1])
    for i in tqdm(range(nbook), desc="libro"):
        photo, c, *_ = synthbook.sample(seenb, rng)
        w.add("train", "synthbook", f"{i:05d}", photo, c)


def chesscog_real(w: Writer, images: Path, ann: Path):
    for name, c in json.loads(ann.read_text()).items():
        w.add("test", "chesscogreal", name.replace("/", "_")[:-4], rgb_of(images / name), np.array(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/corners"))
    ap.add_argument("--video-every", type=int, default=5)
    ap.add_argument("--chesscog-every", type=int, default=2)
    ap.add_argument("--n2d", type=int, default=1500)
    ap.add_argument("--nbook", type=int, default=1500)
    a = ap.parse_args()
    w = Writer(a.out)
    try:
        chessred(w, Path("C:/Users/MartinM/ml_fotos/chessred/annotations.json"), Path("data/raw/chessred"))
        surawut(w, Path("data/raw/surawut_corners/z"))
        user_videos(w, a.video_every)
        samryan(w, Path("data/raw/samryan18"), Path("data/prepared_samryan"))
        chesscog_real(w, Path("data/raw/chesscog_real"), Path("eval_annotations/chesscog_real_corners.json"))
        chesscog_synth(w, Path("data/raw/extracted"), a.chesscog_every)
        synthetic(w, a.n2d, a.nbook)
    finally:
        w.close()


if __name__ == "__main__":
    main()
