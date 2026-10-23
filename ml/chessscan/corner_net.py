"""Detector de esquinas del tablero: mapa de calor estilo CenterNet sobre MobileNetV3-Small.

Entrada 256×256 RGB 0–255 (la imagen entera reducida con «letterbox»). Salidas, con
stride 4 (64×64): `heat` (logit de «acá hay una esquina exterior del área de juego»,
una sola clase: el orden TL/TR/BR/BL se arma después con las 4 mejores) y `off`
(desplazamiento subpíxel dentro de la celda). Ver CONTRACT_CORNERS.md.

  python -m chessscan.corner_net train --run runs/corners_v1 --epochs 30 --steps 500
  python -m chessscan.corner_net eval --run runs/corners_v1
  python -m chessscan.corner_net export --run runs/corners_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np
import tensorflow as tf

IN = 256
STRIDE = 4
HM = IN // STRIDE
SIGMA = 1.5  # en celdas del mapa de calor


# ---------------------------------------------------------------- datos

def load_index(root: Path, split: str) -> list[dict]:
    return [json.loads(l) for l in (root / f"{split}.jsonl").read_text().splitlines()]


def letterbox_matrix(w: int, h: int) -> np.ndarray:
    s = IN / max(w, h)
    return np.array([[s, 0, (IN - w * s) / 2], [0, s, (IN - h * s) / 2], [0, 0, 1]])


def random_view(img, corners, rng):
    """Vista aumentada 256×256: giro libre, escala, inclinación y corrimiento; a veces sale una esquina."""
    h, w = img.shape[:2]
    c = corners.mean(0)
    side = np.linalg.norm(corners - np.roll(corners, -1, 0), axis=1).mean()
    target = IN * rng.uniform(0.3, 0.95)
    s = target / side
    a = rng.uniform(-np.pi, np.pi) if rng.random() < 0.7 else rng.normal(0, 0.2)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * s
    shift = rng.normal(0, 0.12 if rng.random() < 0.85 else 0.3, 2) * IN
    A = np.eye(3)
    A[:2, :2] = R
    A[:2, 2] = -R @ c + IN / 2 + shift
    P = np.eye(3)
    P[2, :2] = rng.normal(0, 0.0006, 2)  # perspectiva extra
    T = np.array([[1, 0, IN / 2], [0, 1, IN / 2], [0, 0, 1]]) @ P @ np.array([[1, 0, -IN / 2], [0, 1, -IN / 2], [0, 0, 1]]) @ A
    bg = tuple(int(v) for v in rng.integers(0, 256, 3))
    out = cv2.warpPerspective(img, T, (IN, IN), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    pc = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float64), T).reshape(4, 2)
    return out, pc


def photometric(img, rng):
    x = img.astype(np.float32)
    x *= rng.uniform(0.6, 1.4) * rng.uniform(0.85, 1.15, 3)
    m = x.mean()
    x = (x - m) * rng.uniform(0.6, 1.4) + m
    x = np.clip(x, 0, 255)
    x = 255 * (x / 255) ** rng.uniform(0.7, 1.4)
    img = x.astype(np.uint8)
    if rng.random() < 0.4:  # oclusiones: manos, brazos, objetos
        for _ in range(int(rng.integers(1, 4))):
            col = tuple(int(v) for v in (np.array([200, 150, 120]) * rng.uniform(0.4, 1.2) if rng.random() < 0.6 else rng.integers(0, 256, 3)))
            cx, cy = rng.integers(0, IN, 2)
            ax = (int(rng.integers(10, 70)), int(rng.integers(8, 40)))
            cv2.ellipse(img, (int(cx), int(cy)), ax, float(rng.uniform(0, 180)), 0, 360, col, -1)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.5))
    if rng.random() < 0.4:
        img = np.clip(img + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


def targets(pc: np.ndarray):
    heat = np.zeros((HM, HM), np.float32)
    off = np.zeros((HM, HM, 2), np.float32)
    mask = np.zeros((HM, HM), np.float32)
    yy, xx = np.mgrid[0:HM, 0:HM].astype(np.float32)
    for x, y in pc / STRIDE:
        if not (0 <= x < HM and 0 <= y < HM):
            continue
        heat = np.maximum(heat, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * SIGMA ** 2)))
        ix, iy = int(x), int(y)
        heat[iy, ix] = 1.0
        off[iy, ix] = (x - ix, y - iy)
        mask[iy, ix] = 1.0
    return heat[..., None], np.concatenate([off, mask[..., None]], -1)


class Batches(keras.utils.PyDataset):
    def __init__(self, root: Path, items: list[dict], bs: int, steps: int, seed: int, weights: dict, workers=8):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=16)
        self.root, self.items, self.bs, self.steps, self.seed = root, items, bs, steps, seed
        w = np.array([weights.get(it["src"], 1.0) for it in items])
        self.p = w / w.sum()
        self.epoch = 0

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        self.epoch += 1

    def __getitem__(self, k):
        rng = np.random.default_rng([self.seed, self.epoch, k])
        xs, hs, os_ = [], [], []
        for j in rng.choice(len(self.items), self.bs, p=self.p):
            it = self.items[j]
            img = cv2.cvtColor(cv2.imread(str(self.root / it["img"])), cv2.COLOR_BGR2RGB)
            v, pc = random_view(img, np.array(it["corners"], float), rng)
            v = photometric(v, rng)
            h, o = targets(pc)
            xs.append(v)
            hs.append(h)
            os_.append(o)
        return np.stack(xs).astype(np.float32), {"heat": np.stack(hs), "off": np.stack(os_)}


# ---------------------------------------------------------------- modelo

def build(width: int = 64) -> keras.Model:
    inp = keras.Input((IN, IN, 3), name="image")
    base = keras.applications.MobileNetV3Small(input_shape=(IN, IN, 3), include_top=False, weights="imagenet",
                                               include_preprocessing=True)
    feats = {}
    for l in base.layers:  # última capa con cada tamaño espacial (no las de padding)
        s = l.output.shape
        if len(s) == 4 and s[1] in (32, 16, 8) and "pad" not in l.name:
            feats[s[1]] = l.output
    fe = keras.Model(base.input, [feats[32], feats[16], feats[8]])
    c3, c4, c5 = fe(inp)
    p = keras.layers.Conv2D(width, 1, activation="relu")(c5)
    p = keras.layers.UpSampling2D(2, interpolation="bilinear")(p)
    p = keras.layers.Add()([p, keras.layers.Conv2D(width, 1, activation="relu")(c4)])
    p = keras.layers.UpSampling2D(2, interpolation="bilinear")(p)
    p = keras.layers.Add()([p, keras.layers.Conv2D(width, 1, activation="relu")(c3)])
    p = keras.layers.SeparableConv2D(width, 3, padding="same", activation="relu")(p)
    p = keras.layers.UpSampling2D(2, interpolation="bilinear")(p)
    p = keras.layers.SeparableConv2D(width // 2, 3, padding="same", activation="relu")(p)
    heat = keras.layers.Conv2D(1, 1, name="heat", bias_initializer=keras.initializers.Constant(-4.0))(p)
    off = keras.layers.Conv2D(2, 1, name="off")(p)
    return keras.Model(inp, {"heat": heat, "off": off}, name="corner_net")


def focal(y, logit):
    p = tf.clip_by_value(tf.sigmoid(logit), 1e-4, 1 - 1e-4)
    pos = tf.cast(y >= 0.999, tf.float32)
    pl = -pos * (1 - p) ** 2 * tf.math.log(p)
    nl = -(1 - pos) * (1 - y) ** 4 * p ** 2 * tf.math.log(1 - p)
    n = tf.maximum(tf.reduce_sum(pos), 1.0)
    return (tf.reduce_sum(pl) + tf.reduce_sum(nl)) / n


def off_loss(y, pred):
    m = y[..., 2:3]
    return tf.reduce_sum(tf.abs(pred - y[..., :2]) * m) / tf.maximum(tf.reduce_sum(m), 1.0)


# ---------------------------------------------------------------- decodificación y métricas

def decode(heat_logit: np.ndarray, off: np.ndarray, k: int = 4, thr: float = 0.15):
    """Las k mejores esquinas (en coordenadas de la entrada de 256) y sus puntajes."""
    h = 1 / (1 + np.exp(-heat_logit[..., 0]))
    mx = cv2.dilate(h, np.ones((5, 5), np.uint8))
    peaks = np.argwhere((h == mx) & (h > thr))
    order = np.argsort(-h[peaks[:, 0], peaks[:, 1]])[:k]
    pts, sc = [], []
    for y, x in peaks[order]:
        dx, dy = off[y, x]
        pts.append(((x + dx) * STRIDE, (y + dy) * STRIDE))
        sc.append(float(h[y, x]))
    return np.array(pts), np.array(sc)


def corner_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Error máximo por esquina, en casillas, con la mejor asignación entre predichas y verdaderas."""
    if len(pred) < 4:
        return float("inf")
    side = np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean() / 8
    best = min(max(np.linalg.norm(pred[list(perm)] - gt, axis=1)) for perm in itertools.permutations(range(4)))
    return best / side


def predict_full(predict, img: np.ndarray):
    """Imagen entera → 4 esquinas en píxeles de la imagen (o menos si no las encuentra)."""
    h, w = img.shape[:2]
    M = letterbox_matrix(w, h)
    x = cv2.warpAffine(img, M[:2], (IN, IN), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    out = predict(x[None].astype(np.float32))
    pts, sc = decode(np.asarray(out["heat"][0]), np.asarray(out["off"][0]))
    if len(pts) == 0:
        return pts, sc
    pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(M)).reshape(-1, 2)
    return pts, sc


def predict_zoom(predict, img: np.ndarray, margin: float = 0.3):
    """Dos pasadas: imagen entera y después un recorte cuadrado alrededor del tablero (lado = (1 + margin) × el
    mayor del rectángulo de las 4 esquinas), para ganar resolución cuando el tablero ocupa poco de la imagen.
    Cada esquina de la 1.ª pasada se reemplaza por la más cercana de la 2.ª, si está a menos de 1 casilla."""
    pts, sc = predict_full(predict, img)
    if len(pts) < 4:
        return pts, sc
    lo, hi = pts.min(0), pts.max(0)
    side = (1 + margin) * (hi - lo).max()
    c = (lo + hi) / 2
    s = IN / side
    M = np.array([[s, 0, IN / 2 - s * c[0]], [0, s, IN / 2 - s * c[1]], [0, 0, 1]])
    x = cv2.warpAffine(img, M[:2], (IN, IN), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    out = predict(x[None].astype(np.float32))
    p2, sc2 = decode(np.asarray(out["heat"][0]), np.asarray(out["off"][0]))
    if len(p2) == 0:
        return pts, sc
    p2 = cv2.perspectiveTransform(p2.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(M)).reshape(-1, 2)
    sq = (hi - lo).max() / 8
    pts = pts.copy()
    for k in range(len(pts)):
        d = np.linalg.norm(p2 - pts[k], axis=1)
        if d.min() < sq:
            pts[k] = p2[d.argmin()]
    return pts, sc


# ---------------------------------------------------------------- main

WEIGHTS = {"chessred": 2.0, "surawut": 0.8, "video1": 1.0, "video3": 1.5, "video2": 0.4, "video6book": 0.6,
           "samryan": 0.7, "chesscog": 1.0, "synth2d": 0.6, "synthbook": 0.8}


def train(a):
    keras.utils.set_random_seed(a.seed)
    a.run.mkdir(parents=True, exist_ok=True)
    tr, va = load_index(a.data, "train"), load_index(a.data, "val")
    model = build()
    sched = keras.optimizers.schedules.CosineDecay(1e-5, a.epochs * a.steps, warmup_target=a.lr, warmup_steps=500)
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
                  loss={"heat": focal, "off": off_loss}, loss_weights={"heat": 1.0, "off": 1.0})
    vb = Batches(a.data, va, a.batch, 40, seed=999, weights={}, workers=4)
    model.fit(Batches(a.data, tr, a.batch, a.steps, a.seed, WEIGHTS), epochs=a.epochs, validation_data=vb,
              callbacks=[keras.callbacks.BackupAndRestore(str(a.run / "backup")),
                         keras.callbacks.ModelCheckpoint(a.run / "model.keras", monitor="val_loss", save_best_only=True),
                         keras.callbacks.CSVLogger(a.run / "history.csv", append=True)], verbose=2)


def evaluate(a):
    model = keras.models.load_model(a.run / "model.keras", custom_objects={"focal": focal, "off_loss": off_loss})
    predict = tf.function(lambda x: model(x, training=False))
    res = {}
    for split in ("val", "test"):
        items = load_index(a.data, split)
        by = {}
        for it in items:
            by.setdefault(it["src"], []).append(it)
        for src, its in sorted(by.items()):
            errs, ms = [], []
            for it in its[:: max(1, len(its) // a.max_per_src)]:
                img = cv2.cvtColor(cv2.imread(str(a.data / it["img"])), cv2.COLOR_BGR2RGB)
                t = time.perf_counter()
                pts, _ = (predict_zoom if a.zoom else predict_full)(predict, img)
                ms.append((time.perf_counter() - t) * 1000)
                errs.append(corner_error(pts, np.array(it["corners"])))
            e = np.array(errs)
            r = {"n": len(e), "exito_0.1": float((e < 0.1).mean()), "exito_0.25": float((e < 0.25).mean()),
                 "exito_0.5": float((e < 0.5).mean()), "error_mediano_casillas": float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else None,
                 "ms_mediana": float(np.median(ms))}
            res[f"{split}/{src}"] = r
            print(f"{split:5s} {src:13s} n={r['n']:4d}  <0,1: {r['exito_0.1']:.3f}  <0,25: {r['exito_0.25']:.3f}  <0,5: {r['exito_0.5']:.3f}  mediana {r['error_mediano_casillas']}", flush=True)
    (a.run / ("eval_zoom.json" if a.zoom else "eval.json")).write_text(json.dumps(res, indent=1))


def export(a):
    model = keras.models.load_model(a.run / "model.keras", custom_objects={"focal": focal, "off_loss": off_loss})
    sm = a.run / "saved_model"
    arch = keras.export.ExportArchive()
    arch.track(model)
    arch.add_endpoint(name="serve", fn=lambda x: model(x, training=False),
                      input_signature=[tf.TensorSpec([1, IN, IN, 3], tf.float32, name="image")])
    arch.write_out(str(sm))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm), signature_keys=["serve"])
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    (a.run / "corner_detector_fp16.tflite").write_bytes(conv.convert())
    print("ok", (a.run / "corner_detector_fp16.tflite").stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "export"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/corners"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-per-src", type=int, default=300)
    ap.add_argument("--zoom", action="store_true", help="eval: segunda pasada con zoom sobre el tablero")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "export": export}[a.cmd](a)


if __name__ == "__main__":
    main()
