"""Generador paramétrico de juegos de piezas tipo Staunton (mallas OBJ), para el render sintético.

Cada «juego» sale de una semilla: proporciones generales (base, cintura, alto,
collares), estilo de cada pieza (corona de la dama, almenas de la torre, mitra del
alfil, cruz o bola del rey) y silueta del caballo. Peón, torre, alfil, dama y rey
son superficies de revolución de un perfil (r, z) más agregados simples; el
caballo es la silueta extruida sobre una base de revolución. Unidades: alto del
rey = 1. Los materiales y colores se asignan en el render (Blender), no acá.

  python -m chessscan.staunton --out data/raw/staunton --sets 200 --seed 0 [--preview prev.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PIECES = ["P", "N", "B", "R", "Q", "K"]


class Mesh:
    def __init__(self):
        self.v: list[np.ndarray] = []
        self.f: list[np.ndarray] = []
        self.n = 0

    def add(self, v: np.ndarray, f: np.ndarray) -> None:
        self.v.append(v)
        self.f.append(f + self.n)
        self.n += len(v)

    def write_obj(self, path: Path) -> None:
        v, f = np.concatenate(self.v), np.concatenate(self.f)
        with open(path, "w") as fh:
            fh.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in v))
            fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f))


def lathe(profile: np.ndarray, seg: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Superficie de revolución de un perfil [(r, z)] alrededor del eje z (cerrada abajo y arriba)."""
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    r, z = profile[:, 0], profile[:, 1]
    v = np.stack([np.outer(r, np.cos(th)), np.outer(r, np.sin(th)), np.repeat(z[:, None], seg, 1)], -1).reshape(-1, 3)
    f = []
    n = len(profile)
    for i in range(n - 1):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            c, d = a + seg, b + seg
            f += [(a, b, d), (a, d, c)]
    # Tapas.
    v = np.concatenate([v, [[0, 0, z[0]], [0, 0, z[-1]]]])
    bot, top = len(v) - 2, len(v) - 1
    for j in range(seg):
        f.append((bot, (j + 1) % seg, j))
        f.append((top, (n - 1) * seg + j, (n - 1) * seg + (j + 1) % seg))
    return v, np.array(f)


def box(c, s) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.asarray(c, float), np.asarray(s, float) / 2
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
    f = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                  (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])
    return v, f


def sphere(c, r, seg=16):
    t = np.linspace(0, np.pi, seg // 2 + 1)
    prof = np.stack([np.sin(t) * r, -np.cos(t) * r], 1)
    v, f = lathe(prof, seg)
    return v + np.asarray(c), f


def extrude(poly: np.ndarray, depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Prisma de una silueta 2D (x, z) con espesor en y. Triangula en abanico desde el centroide."""
    n = len(poly)
    cx, cz = poly.mean(0)
    front = np.c_[poly[:, 0], np.full(n, -depth / 2), poly[:, 1]]
    back = np.c_[poly[:, 0], np.full(n, depth / 2), poly[:, 1]]
    v = np.concatenate([front, back, [[cx, -depth / 2, cz], [cx, depth / 2, cz]]])
    cf, cb = 2 * n, 2 * n + 1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j), (i, n + j, n + i), (cf, j, i), (cb, n + i, n + j)]
    return v, np.array(f)


class Style:
    """Proporciones del juego (comunes a todas las piezas) a partir de una semilla."""

    def __init__(self, rng: np.random.Generator):
        self.base_r = rng.uniform(0.17, 0.24)        # radio de la base relativo al alto del rey
        self.base_h = rng.uniform(0.05, 0.10)
        self.waist = rng.uniform(0.35, 0.55)         # radio de la cintura / radio de base
        self.collar = rng.uniform(0.75, 1.05)        # radio del collar / radio de base
        self.heights = {"K": 1.0, "Q": rng.uniform(0.86, 0.94), "B": rng.uniform(0.70, 0.80),
                        "N": rng.uniform(0.62, 0.72), "R": rng.uniform(0.52, 0.62), "P": rng.uniform(0.42, 0.52)}
        self.pawn_head = rng.uniform(0.10, 0.14)
        self.rook_crenels = int(rng.choice([4, 5, 6, 8]))
        self.queen_points = int(rng.choice([0, 7, 8, 9, 10, 12]))  # 0 = corona lisa con bola
        self.king_top = rng.choice(["cross", "cross", "cross_pattee", "ball"])
        self.bishop_slit = bool(rng.random() < 0.7)
        self.knight = rng.uniform(0, 1, 6)           # parámetros de la silueta del caballo
        self.wobble = rng.uniform(0.0, 0.03)          # irregularidad de torneado


def base_profile(st: Style, h: float, rng, top_r: float, top_z: float) -> list[tuple[float, float]]:
    """Base, cuerpo cónico hasta el collar; devuelve el perfil (r, z) hasta top_z."""
    R = st.base_r * (0.85 + 0.15 * h)
    bh = st.base_h * (0.8 + 0.4 * h)
    p = [(R * 0.98, 0), (R, bh * 0.4), (R * 0.92, bh), (R * 0.78, bh * 1.25), (R * 0.82, bh * 1.5)]
    z_col = top_z
    for t in np.linspace(0.15, 1, 6):
        z = bh * 1.5 + (z_col - bh * 1.5) * t
        r = R * 0.75 + (R * st.waist - R * 0.75) * t ** 0.7
        p.append((r * (1 + rng.uniform(-st.wobble, st.wobble)), z))
    p += [(R * st.collar * 0.8, z_col), (R * st.collar * 0.85, z_col + 0.012), (top_r, z_col + 0.03)]
    return p


def make_piece(kind: str, st: Style, rng) -> Mesh:
    m = Mesh()
    H = st.heights[kind]
    R = st.base_r
    if kind == "P":
        col = H * 0.62
        prof = base_profile(st, H, rng, R * 0.35, col)
        head = st.pawn_head * (1 + rng.uniform(-0.1, 0.1))
        v, f = lathe(np.array(prof))
        m.add(v, f)
        m.add(*sphere((0, 0, H - head), head))
    elif kind == "R":
        col = H * 0.55
        prof = base_profile(st, H, rng, R * 0.6, col)
        top_r = R * rng.uniform(0.75, 0.9)
        prof += [(top_r * 0.85, col + 0.05), (top_r, H - 0.09), (top_r, H - 0.04)]
        v, f = lathe(np.array(prof))
        m.add(v, f)
        n = st.rook_crenels
        for i in range(n):
            a = 2 * np.pi * i / n
            w = 2 * np.pi * top_r / n * 0.55
            m.add(*rot_z(box((top_r * 0.82 * np.cos(a), top_r * 0.82 * np.sin(a), H - 0.02), (top_r * 0.32, w, 0.05)), a))
    elif kind == "B":
        col = H * 0.55
        prof = base_profile(st, H, rng, R * 0.3, col)
        mr = R * rng.uniform(0.42, 0.55)
        tz = H - 0.06
        for t in np.linspace(0, 1, 9):
            prof.append((mr * np.sin(np.pi * (0.15 + 0.85 * t)) ** 0.8 * (1 - 0.6 * t ** 3), col + 0.03 + (tz - col - 0.03) * t))
        v, f = lathe(np.array(prof))
        m.add(v, f)
        m.add(*sphere((0, 0, H - 0.03), 0.03))
    elif kind == "Q":
        col = H * 0.6
        prof = base_profile(st, H, rng, R * 0.3, col)
        cr = R * rng.uniform(0.6, 0.75)
        prof += [(cr * 0.55, col + 0.06), (cr, H - 0.1), (cr * 0.95, H - 0.08), (cr * 0.5, H - 0.07)]
        v, f = lathe(np.array(prof))
        m.add(v, f)
        if st.queen_points:
            for i in range(st.queen_points):
                a = 2 * np.pi * i / st.queen_points
                m.add(*sphere((cr * 0.92 * np.cos(a), cr * 0.92 * np.sin(a), H - 0.085), 0.02))
        m.add(*sphere((0, 0, H - 0.045), 0.035))
    elif kind == "K":
        col = H * 0.58
        prof = base_profile(st, H, rng, R * 0.3, col)
        cr = R * rng.uniform(0.55, 0.7)
        prof += [(cr * 0.6, col + 0.06), (cr, H - 0.2), (cr * 0.9, H - 0.16), (cr * 0.45, H - 0.13)]
        v, f = lathe(np.array(prof))
        m.add(v, f)
        if st.king_top == "ball":
            m.add(*sphere((0, 0, H - 0.08), 0.05))
        else:
            w = 0.035 if st.king_top == "cross" else 0.05
            m.add(*box((0, 0, H - 0.07), (0.03, 0.03, 0.13)))
            m.add(*box((0, 0, H - 0.05), (w * 2.4, 0.03, 0.03)))
    elif kind == "N":
        col = H * 0.32
        prof = base_profile(st, H, rng, R * 0.55, col)
        v, f = lathe(np.array(prof))
        m.add(v, f)
        m.add(*extrude(knight_outline(st, H, col), R * rng.uniform(0.5, 0.75)))
    return m


def rot_z(vf, a):
    v, f = vf
    c, s = np.cos(a), np.sin(a)
    return v @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T, f


def knight_outline(st: Style, H: float, z0: float) -> np.ndarray:
    """Silueta lateral del caballo (x hacia el hocico, z hacia arriba), convexa por tramos."""
    k = st.knight
    R = st.base_r
    neck = R * (0.55 + 0.25 * k[0])
    snout = R * (0.9 + 0.5 * k[1])
    top = H * (0.97 + 0.03 * k[2])
    ear = 0.05 + 0.04 * k[3]
    jaw = z0 + (top - z0) * (0.45 + 0.15 * k[4])
    back = -R * (0.6 + 0.25 * k[5])
    pts = [(back, z0), (neck, z0), (neck * 0.9, jaw * 0.98), (snout, jaw), (snout * 1.02, jaw + 0.06),
           (snout * 0.7, top - 0.12), (R * 0.15, top - 0.02), (0.0, top + ear), (-R * 0.15, top - 0.03),
           (back * 0.9, top - 0.2), (back * 1.05, (top + z0) / 2)]
    return np.array(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/staunton"))
    ap.add_argument("--sets", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=Path, help="PNG con las siluetas de los primeros juegos")
    a = ap.parse_args()
    meta = []
    for s in range(a.sets):
        rng = np.random.default_rng([a.seed, s])
        st = Style(rng)
        d = a.out / f"set{s:04d}"
        d.mkdir(parents=True, exist_ok=True)
        for k in PIECES:
            make_piece(k, st, rng).write_obj(d / f"{k}.obj")
        meta.append({"set": d.name, "seed": [a.seed, s], **{k: (v.item() if hasattr(v, "item") else v)
                                                           for k, v in vars(st).items() if k not in ("knight", "heights")},
                     "heights": st.heights})
    (a.out / "sets.json").write_text(json.dumps(meta, indent=1, default=str))
    if a.preview:
        preview(a.out, min(8, a.sets), a.preview)
    print(f"{a.sets} juegos en {a.out}")


def preview(root: Path, n: int, path: Path) -> None:
    """Proyección lateral de cada pieza (silueta) para revisar a ojo, sin render."""
    import cv2

    rows = []
    for s in range(n):
        tiles = []
        for k in PIECES:
            v = []
            for line in open(root / f"set{s:04d}" / f"{k}.obj"):
                if line.startswith("v "):
                    v.append([float(t) for t in line.split()[1:]])
            v = np.array(v)
            img = np.full((140, 90), 255, np.uint8)
            fv = []
            for line in open(root / f"set{s:04d}" / f"{k}.obj"):
                if line.startswith("f "):
                    fv.append([int(t) - 1 for t in line.split()[1:]])
            for tri in fv:
                p = np.int32([[45 + v[i, 0] * 110, 132 - v[i, 2] * 125] for i in tri])
                cv2.fillConvexPoly(img, p, 60)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(path), np.vstack(rows))


if __name__ == "__main__":
    main()
