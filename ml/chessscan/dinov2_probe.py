"""DINOv2 ViT-S/14 congelado + cabeza lineal o MLP, con el recorte del contrato v1.

Corre en un entorno con torch y transformers (no en el de TensorFlow). El recorte
de 128×64 (alto × ancho) se lleva a 224×112, múltiplo de 14 (16 × 8 parches),
con la normalización de ImageNet. Features = CLS ‖ media de parches (768).

  python -m chessscan.dinov2_probe features --out runs/dino --train-batches 400
  python -m chessscan.dinov2_probe probe --out runs/dino
  python -m chessscan.dinov2_probe flops
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import geometry as g
from .data import Boards, TrainBatches

H, W = 224, 112
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Mismas fuentes que el v5 (sin los videos de test).
TRAIN = [("data/prepared/train", 1.0), ("data/real/video1", 0.5), ("data/real/video3", 0.8), ("data/real/video6book_train", 0.3),
         ("data/preparedbook/train", 0.35), ("data/preparedbook_b/train", 0.3), ("data/prepared2d_tex/train", 0.3),
         ("data/prepared2d_tex_b/train", 0.2), ("data/prepared2d/train", 0.15), ("data/prepared_chessred/train", 0.8),
         ("data/prepared_samryan/train", 0.2)]
TESTS = {"video4": ("data/real/video4", None), "video5": ("data/real/video5", None),
         "libro_monitor": ("data/real/video6book", (1000, 1412))}


def load_model():
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    return m


@torch.no_grad()
def embed(model, crops_u8: np.ndarray, bs: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(crops_u8), bs):
        x = torch.from_numpy(crops_u8[i:i + bs]).permute(0, 3, 1, 2).float() / 255
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = (x - MEAN) / STD
        h = model(pixel_values=x).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy().astype(np.float16))
    return np.concatenate(out)


def test_crops(path: str, rng_, every: int):
    d = Path(path)
    b = Boards.load(d)
    names = (d / "names.txt").read_text().split()
    idx = [k for k, n in enumerate(names) if rng_ is None or rng_[0] <= int(n[1:]) <= rng_[1]][::every]
    x = np.concatenate([g.crop_all(np.asarray(b.canon[k])) for k in idx])
    y = b.labels[idx].reshape(-1)
    keep = y >= 0
    return x[keep], y[keep]


def features(a) -> None:
    torch.set_num_threads(a.threads)
    model = load_model()
    a.out.mkdir(parents=True, exist_ok=True)
    tb = TrainBatches([(Boards.load(Path(p)), w) for p, w in TRAIN], 128, a.train_batches, seed=1234)
    t = time.time()
    xs, ys = [], []
    for k in range(a.train_batches):
        x, y = tb.batch(k)
        xs.append(embed(model, x.astype(np.uint8)))
        ys.append(y)
        if k % 20 == 0:
            print(f"train {k}/{a.train_batches} {time.time() - t:.0f} s", flush=True)
    np.savez(a.out / "train.npz", x=np.concatenate(xs), y=np.concatenate(ys))
    for name, (p, r) in TESTS.items():
        x, y = test_crops(p, r, a.every)
        np.savez(a.out / f"{name}.npz", x=embed(model, x), y=y)
        print(name, len(y), f"{time.time() - t:.0f} s", flush=True)


def probe(a) -> None:
    torch.manual_seed(0)
    tr = np.load(a.out / "train.npz")
    X, Y = torch.from_numpy(tr["x"].astype(np.float32)), torch.from_numpy(tr["y"])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    res = {}
    for kind in ("lineal", "mlp"):
        head = torch.nn.Linear(768, 13) if kind == "lineal" else torch.nn.Sequential(
            torch.nn.Linear(768, 512), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, 13))
        opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-4)
        for ep in range(30):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 512):
                j = perm[i:i + 512]
                loss = torch.nn.functional.cross_entropy(head((X[j] - mu) / sd), Y[j])
                opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        for name in TESTS:
            te = np.load(a.out / f"{name}.npz")
            with torch.no_grad():
                p = head((torch.from_numpy(te["x"].astype(np.float32)) - mu) / sd).argmax(1).numpy()
            y = te["y"]
            occ = y != 0
            res[f"{kind}/{name}"] = {"casillas": float((p == y).mean()), "piezas": float((p[occ] == y[occ]).mean()), "n": int(len(y))}
            print(kind, name, res[f"{kind}/{name}"], flush=True)
    (a.out / "probe.json").write_text(json.dumps(res, indent=1))


def flops(a) -> None:
    """FLOPs de DINOv2 ViT-S/14 por recorte de 224×112 (128 parches + CLS) y por tablero de 64."""
    d, L, n = 384, 12, 1 + (H // 14) * (W // 14)
    per_layer = 2 * n * (4 * d * d + 2 * 4 * d * d) + 2 * 2 * n * n * d  # QKV+proy, MLP 4×, atención
    total = L * per_layer + 2 * (H // 14) * (W // 14) * (14 * 14 * 3) * d
    print(json.dumps({"tokens": n, "gflops_por_recorte": total / 1e9, "gflops_por_tablero_64": 64 * total / 1e9,
                      "params_M": 22.1, "fp16_MB": 22.1 * 2, "int8_MB": 22.1}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "probe", "flops"])
    ap.add_argument("--out", type=Path, default=Path("runs/dino"))
    ap.add_argument("--train-batches", type=int, default=400)
    ap.add_argument("--every", type=int, default=10, help="1 de cada N cuadros de test")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    {"features": features, "probe": probe, "flops": flops}[a.cmd](a)


if __name__ == "__main__":
    main()
