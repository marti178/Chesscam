"""Generador sintético de tableros 2D en pantalla (Lichess, chess.com…) fotografiados.

Cada muestra: posición aleatoria → tablero plano (tema, set de piezas, resaltados,
jaque, coordenadas, flechas) → captura de pantalla con interfaz alrededor →
«foto» del monitor (rejilla de subpíxeles/moiré, perspectiva, brillo, reflejo,
desenfoque, ruido, JPEG) → imagen canónica según CONTRACT v1.

Salida en el mismo formato que prepare_chesscog (canon.npy, labels.npy, names.txt).

  python -m chessscan.synth2d --pieces data/pieces_png --out data/prepared2d --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g
from .pieces2d import PIECES, TEST_SETS

# (clara, oscura) en RGB. Temas de Lichess y chess.com más comunes y algunos genéricos.
THEMES = [
    ((240, 217, 181), (181, 136, 99)),   # lichess brown
    ((222, 227, 230), (140, 162, 173)),  # lichess blue
    ((255, 255, 221), (134, 166, 102)),  # lichess green
    ((238, 238, 210), (118, 150, 86)),   # chess.com green
    ((237, 214, 176), (184, 135, 98)),   # chess.com brown
    ((234, 233, 210), (75, 115, 153)),   # chess.com blue
    ((240, 241, 240), (196, 216, 228)),  # chess.com ice/sky
    ((232, 235, 239), (125, 135, 150)),  # gris
    ((229, 218, 240), (159, 144, 176)),  # violeta
    ((241, 222, 222), (186, 85, 85)),    # rojo
    ((206, 206, 206), (120, 120, 120)),  # gris neutro
]
HIGHLIGHTS = [((155, 199, 0), 0.41), ((247, 247, 105), 0.5), ((255, 255, 51), 0.45), ((20, 85, 30), 0.5)]
ARROWS = [(21, 120, 27), (136, 32, 32), (0, 48, 136), (230, 143, 0)]
SYMBOLS = {"K": "K", "Q": "Q", "R": "R", "B": "B", "N": "N", "P": "P"}


def load_sets(root: Path, split: str) -> dict[str, dict[str, np.ndarray]]:
    sets = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d.name in TEST_SETS) != (split == "heldout"):
            continue
        sets[d.name] = {p: cv2.cvtColor(cv2.imread(str(d / f"{p}.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                        for p in PIECES}
    return sets


def random_board(rng) -> list[list[str | None]]:
    """Posición aleatoria (no necesariamente legal), con densidad variable."""
    b = [[None] * 8 for _ in range(8)]  # b[rank][file], rank 0 = fila 1
    n = int(rng.integers(2, 33))
    squares = rng.permutation(64)[:n]
    pool = ["K", "Q", "R", "R", "B", "B", "N", "N"] + ["P"] * 8
    kinds = []
    for color in "wb":
        kinds += [color + "K"]
    while len(kinds) < n:
        kinds.append(("w" if rng.random() < 0.5 else "b") + pool[int(rng.integers(1, len(pool)))])
    for sq, k in zip(squares, kinds):
        r, f = divmod(int(sq), 8)
        if k[1] == "P" and r in (0, 7):
            r = int(rng.integers(1, 7))
            if b[r][f] is not None:
                continue
        b[r][f] = k
    return b


def to_fen(b) -> str:
    rows = []
    for r in range(7, -1, -1):
        s, e = "", 0
        for f in range(8):
            p = b[r][f]
            if p is None:
                e += 1
                continue
            if e:
                s, e = s + str(e), 0
            s += p[1] if p[0] == "w" else p[1].lower()
        rows.append(s + (str(e) if e else ""))
    return "/".join(rows)


def alpha_paste(dst: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = src[..., 3:] / 255
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (src[..., :3] * a + roi * (1 - a)).astype(np.uint8)


def render_board(b, pieces, rng, sq: int, flipped: bool) -> np.ndarray:
    light, dark = THEMES[int(rng.integers(len(THEMES)))]
    jit = rng.integers(-12, 13, size=3)
    light = np.clip(np.array(light) + jit, 0, 255)
    dark = np.clip(np.array(dark) + jit, 0, 255)
    B = 8 * sq
    img = np.zeros((B, B, 3), np.uint8)
    for row in range(8):
        for col in range(8):
            img[row * sq:(row + 1) * sq, col * sq:(col + 1) * sq] = light if (row + col) % 2 == 0 else dark
    if rng.random() < 0.25:  # textura tipo madera/mármol
        tex = cv2.GaussianBlur(rng.normal(0, 1, (B // 4, B // 4)).astype(np.float32), (0, 0), 2)
        tex = cv2.resize(tex, (B, B))[..., None]
        img = np.clip(img + tex * rng.uniform(4, 15), 0, 255).astype(np.uint8)

    def cell(rank, file):  # (fila, col) en pantalla
        return (7 - rank, file) if not flipped else (rank, 7 - file)

    # Última jugada resaltada.
    if rng.random() < 0.6:
        color, a = HIGHLIGHTS[int(rng.integers(len(HIGHLIGHTS)))]
        for _ in range(2):
            r, c = int(rng.integers(8)), int(rng.integers(8))
            roi = img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq].astype(np.float32)
            img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = (roi * (1 - a) + np.array(color) * a).astype(np.uint8)
    # Jaque: halo rojo radial debajo de un rey.
    kings = [(rk, fl) for rk in range(8) for fl in range(8) if b[rk][fl] and b[rk][fl][1] == "K"]
    if kings and rng.random() < 0.2:
        rk, fl = kings[int(rng.integers(len(kings)))]
        r, c = cell(rk, fl)
        yy, xx = np.mgrid[0:sq, 0:sq].astype(np.float32)
        d = np.sqrt((yy - sq / 2) ** 2 + (xx - sq / 2) ** 2) / (sq / 2)
        a = np.clip(1 - d, 0, 1)[..., None] * 0.9
        roi = img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq].astype(np.float32)
        img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = (roi * (1 - a) + np.array([255, 0, 0]) * a).astype(np.uint8)
    # Coordenadas dentro de las casillas del borde.
    if rng.random() < 0.7:
        fs = sq / 110
        for i in range(8):
            files = "abcdefgh" if not flipped else "hgfedcba"
            ranks = "87654321" if not flipped else "12345678"
            col_txt = tuple(int(v) for v in (dark if (7 + i) % 2 == 0 else light))
            cv2.putText(img, files[i], (i * sq + sq - int(sq * 0.22), B - int(sq * 0.06)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col_txt, max(1, sq // 45), cv2.LINE_AA)
            col_txt = tuple(int(v) for v in (dark if i % 2 == 0 else light))
            cv2.putText(img, ranks[i], (int(sq * 0.05), i * sq + int(sq * 0.28)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col_txt, max(1, sq // 45), cv2.LINE_AA)
    # Piezas.
    scale = rng.uniform(0.78, 1.0)
    ps = max(8, int(sq * scale))
    cache = {}
    for rk in range(8):
        for fl in range(8):
            p = b[rk][fl]
            if p is None:
                continue
            if p not in cache:
                cache[p] = cv2.resize(pieces[p], (ps, ps), interpolation=cv2.INTER_AREA)
            r, c = cell(rk, fl)
            off = (sq - ps) // 2
            alpha_paste(img, cache[p], c * sq + off, r * sq + off + (int(rng.integers(0, max(1, off + 1))) if rng.random() < 0.1 else 0))
    # Flechas (encima de las piezas, semitransparentes).
    for _ in range(int(rng.choice([0, 0, 0, 1, 1, 2]))):
        over = img.copy()
        color = tuple(int(v) for v in ARROWS[int(rng.integers(len(ARROWS)))])
        p0 = (int(rng.integers(8)) * sq + sq // 2, int(rng.integers(8)) * sq + sq // 2)
        p1 = (int(rng.integers(8)) * sq + sq // 2, int(rng.integers(8)) * sq + sq // 2)
        if p0 != p1:
            cv2.arrowedLine(over, p0, p1, color, max(2, sq // 6), cv2.LINE_AA, tipLength=0.25 * sq / max(1, np.hypot(p1[0] - p0[0], p1[1] - p0[1])) * 2)
            img = cv2.addWeighted(over, 0.65, img, 0.35, 0)
    return img


def screenshot(board: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Tablero dentro de una interfaz; devuelve (imagen, esquinas del tablero)."""
    B = board.shape[0]
    W = int(B * rng.uniform(1.15, 2.2))
    H = int(B * rng.uniform(1.1, 1.5))
    bg = np.array([22, 21, 18]) if rng.random() < 0.6 else rng.integers(0, 256, 3)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = bg
    for _ in range(int(rng.integers(0, 8))):  # paneles y texto de interfaz
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        w, h = int(rng.integers(20, W // 2)), int(rng.integers(10, H // 4))
        cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(v) for v in rng.integers(0, 256, 3)), -1)
        cv2.putText(img, "1. e4 e5 2. Nf3", (x + 4, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    tuple(int(v) for v in rng.integers(0, 256, 3)), 1, cv2.LINE_AA)
    x0 = int(rng.integers(0, W - B + 1))
    y0 = int(rng.integers(0, H - B + 1))
    img[y0:y0 + B, x0:x0 + B] = board
    corners = np.array([[x0, y0], [x0 + B, y0], [x0 + B, y0 + B], [x0, y0 + B]], np.float64)
    return img, corners


def subpixel_grid(img: np.ndarray, rng) -> tuple[np.ndarray, float]:
    """Simula la rejilla RGB del monitor (de ahí sale el moiré al fotografiar)."""
    k = 3
    up = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    mask = np.ones((k, k, 3), np.float32) * rng.uniform(0.2, 0.5)
    for ch in range(3):
        mask[: k - 1, ch, ch] = 1.0
    tile = np.tile(mask, (img.shape[0], img.shape[1], 1))
    return np.clip(up * tile * rng.uniform(1.3, 1.8), 0, 255).astype(np.uint8), k


def photograph(screen: np.ndarray, corners: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Proyecta la pantalla en una «foto» de celular; devuelve (foto RGB, esquinas)."""
    out_w, out_h = (1280, 960) if rng.random() < 0.6 else (960, 1280)
    scale = 1.0
    if rng.random() < 0.6:
        screen, scale = subpixel_grid(screen, rng)
    cs = corners * scale
    # Esquinas destino del tablero: tamaño, posición, inclinación y giro aleatorios.
    size = rng.uniform(0.35, 0.92) * min(out_w, out_h)
    cx, cy = rng.uniform(0.35, 0.65) * out_w, rng.uniform(0.35, 0.65) * out_h
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64) * size / 2
    tilt = rng.uniform(0, 0.35)  # el borde lejano se ve más corto
    sq[:2, 0] *= 1 - tilt
    sq[:, 1] *= 1 - 0.5 * tilt
    sq += rng.normal(0, size * 0.025, sq.shape)
    ang = np.deg2rad(rng.uniform(-10, 10))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = sq @ R.T + [cx, cy]
    Hm = cv2.getPerspectiveTransform(cs.astype(np.float32), dst.astype(np.float32))
    bg = rng.integers(0, 60, 3)
    photo = cv2.warpPerspective(screen, Hm, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(v) for v in bg))
    x = photo.astype(np.float32)
    # Pantalla: gamma, brillo, tinte; reflejo; viñeteado.
    x = 255 * (x / 255) ** rng.uniform(0.7, 1.5)
    x *= rng.uniform(0.6, 1.3) * rng.uniform(0.85, 1.15, 3)
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    if rng.random() < 0.4:
        gx, gy = rng.uniform(0, out_w), rng.uniform(0, out_h)
        r = rng.uniform(0.1, 0.5) * out_w
        x += (rng.uniform(30, 120) * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r)))[..., None]
    v = 1 - rng.uniform(0, 0.4) * (((xx - out_w / 2) / out_w) ** 2 + ((yy - out_h / 2) / out_h) ** 2) * 2
    x *= v[..., None]
    photo = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.4, 1.8))
    if rng.random() < 0.6:
        photo = np.clip(photo + rng.normal(0, rng.uniform(1, 8), photo.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
    photo = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return photo, dst


def sample(sets, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    name = list(sets)[int(rng.integers(len(sets)))]
    b = random_board(rng)
    flipped = bool(rng.random() < 0.35)
    sq = int(rng.integers(40, 121))
    board = render_board(b, sets[name], rng, sq, flipped)
    screen, corners = screenshot(board, rng)
    photo, img_corners = photograph(screen, corners, rng)
    fen = to_fen(b)
    labels = g.board_labels(fen, "TR" if flipped else "BL")
    return photo, img_corners, labels, fen, name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=Path, default=Path("data/pieces_png"))
    ap.add_argument("--out", type=Path, default=Path("data/prepared2d"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", nargs=4, type=int, default=[4000, 200, 300, 300],
                    metavar=("TRAIN", "VAL", "TEST", "TEST_HELDOUT"))
    ap.add_argument("--preview", type=Path, help="guardar 6 fotos de ejemplo acá")
    a = ap.parse_args()
    seen, held = load_sets(a.pieces, "train"), load_sets(a.pieces, "heldout")
    for k, (split, n) in enumerate(zip(["train", "val", "test", "test_heldout"], a.n)):
        sets = held if split == "test_heldout" else seen
        rng = np.random.default_rng([a.seed, k])
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        canon = np.lib.format.open_memmap(d / "canon.npy", mode="w+", dtype=np.uint8, shape=(n, g.CH, g.CW, 3))
        labels, names = np.zeros((n, 64), np.int64), []
        for i in tqdm(range(n), desc=split):
            photo, corners, lab, fen, setname = sample(sets, rng)
            canon[i] = g.warp(photo, g.homography(corners))
            labels[i] = lab
            names.append(f"{setname} {fen}")
            if a.preview and split == "val" and i < 6:
                a.preview.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(a.preview / f"photo{i}.jpg"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(a.preview / f"canon{i}.png"), cv2.cvtColor(canon[i], cv2.COLOR_RGB2BGR))
        canon.flush()
        np.save(d / "labels.npy", labels)
        (d / "names.txt").write_text("\n".join(names) + "\n")


if __name__ == "__main__":
    main()
