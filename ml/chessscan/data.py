"""Carga de datos preparados (canon.npy + labels.npy) y armado de lotes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as g
from .augment import augment_board


@dataclass
class Boards:
    canon: np.ndarray   # uint8 [N, CH, CW, 3] (memmap)
    labels: np.ndarray  # int64 [N, 64]

    @classmethod
    def load(cls, d: Path) -> "Boards":
        return cls(np.load(d / "canon.npy", mmap_mode="r"), np.load(d / "labels.npy"))

    def __len__(self) -> int:
        return len(self.labels)


def all_crops(b: Boards, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Recortes sin aumentar de todas las casillas: ([n*64,128,64,3] uint8, [n*64])."""
    idx = np.arange(len(b)) if idx is None else idx
    x = np.concatenate([g.crop_all(np.asarray(b.canon[i])) for i in idx])
    return x, b.labels[idx].reshape(-1)


class TrainBatches:
    """Lotes aumentados y deterministas: el lote k de la época e depende solo de (seed, e, k).

    Cada lote toma `boards_per_batch` tableros (mezclando fuentes según su peso),
    los aumenta, recorta las 64 casillas y elige `batch_size` casillas con
    probabilidad ∝ frecuencia_de_clase^-0.5 (para que las vacías no dominen).
    """

    def __init__(self, sources: list[tuple[Boards, float]], batch_size: int, steps: int,
                 seed: int, boards_per_batch: int = 8, geo_px: float = 4.0):
        self.sources = sources
        w = np.array([s[1] for s in sources], float)
        self.src_p = w / w.sum()
        self.batch_size, self.steps, self.seed = batch_size, steps, seed
        self.bpb, self.geo_px = boards_per_batch, geo_px
        lab = np.concatenate([s[0].labels.reshape(-1) for s in sources])
        freq = np.bincount(lab, minlength=len(g.CLASSES)).astype(float) + 1
        self.class_w = freq ** -0.5
        self.epoch = 0

    def __len__(self) -> int:
        return self.steps

    def batch(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([self.seed, self.epoch, k])
        xs, ys = [], []
        for _ in range(self.bpb):
            b = self.sources[rng.choice(len(self.sources), p=self.src_p)][0]
            i = int(rng.integers(len(b)))
            canon, lab = augment_board(np.asarray(b.canon[i]), b.labels[i], rng, self.geo_px)
            xs.append(g.crop_all(canon))
            ys.append(lab)
        x, y = np.concatenate(xs), np.concatenate(ys)
        p = self.class_w[y]
        sel = rng.choice(len(y), size=self.batch_size, replace=False, p=p / p.sum())
        return x[sel].astype(np.float32), y[sel]
