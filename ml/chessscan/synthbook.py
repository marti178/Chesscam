"""Generador sintético de diagramas de libros de ajedrez, fotografiados o escaneados.

Cada muestra: posición aleatoria → diagrama impreso en blanco y negro (casillas
oscuras rayadas a varios ángulos, punteadas, grises lisas o blancas; piezas tipo
figurine de libro con el rayado recortado alrededor; marco simple o doble;
coordenadas a veces) → página con texto y notación figurine alrededor, papel
amarillento con textura, tinta corrida o gastada → «foto» del celular a la página
(curvatura cerca del lomo, perspectiva, sombra de la mano, luz despareja,
desenfoque, JPEG) o «escaneo» (casi de frente, contraste alto) → canónica (CONTRACT v1).

Las piezas salen de los SVG de Lichess ya rasterizados (`chessscan.pieces2d`),
pasados a tinta (blanco y negro). Licencias en LICENSES.md.

  python -m chessscan.synthbook --pieces data/pieces_png --out data/preparedbook --seed 11
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import geometry as g
from .pieces2d import PIECES, TEST_SETS
from .synth2d import alpha_paste, random_board, to_fen

# Sets con aspecto de figurine impresa (los demás son demasiado coloridos o de fantasía).
BOOK_SETS = {"cburnett", "merida", "mpchess", "maestro", "california", "cardinal", "icpieces", "tatiana",
             "gioco", "fresca", "cooke", "dubrovny", "spatial", "fantasy", "celtic", "chessnut", "staunty"}
FONTS = [cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SIMPLEX,
         cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_DUPLEX]


def ink_piece(rgba: np.ndarray, rng) -> np.ndarray:
    """Pieza a tinta: gris → negro o papel (umbral suave), alfa conservada. Devuelve [h,w,2] (tinta, alfa)."""
    gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    t = rng.uniform(90, 160)
    ink = 1 / (1 + np.exp((gray - t) / rng.uniform(4, 12)))  # 1 = tinta
    return np.dstack([ink, rgba[..., 3] / 255.0]).astype(np.float32)


def load_book_sets(root: Path, split: str) -> dict[str, dict[str, np.ndarray]]:
    sets = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name not in BOOK_SETS or (d.name in TEST_SETS) != (split == "heldout"):
            continue
        sets[d.name] = {p: cv2.cvtColor(cv2.imread(str(d / f"{p}.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                        for p in PIECES}
    return sets


def dark_pattern(sq: int, rng) -> np.ndarray:
    """Tinta (0–1) de una casilla oscura de sq×sq."""
    style = rng.choice(["hatch", "hatch", "hatch", "hatch2", "dots", "gray", "white"], p=[.3, .2, .15, .1, .1, .1, .05])
    yy, xx = np.mgrid[0:sq, 0:sq].astype(np.float32)
    if style in ("hatch", "hatch2"):
        ang = np.deg2rad(rng.choice([45, -45, 45, 135]) + rng.uniform(-5, 5)) if style == "hatch" else rng.uniform(0, np.pi)
        period = max(2.2, sq / rng.uniform(5, 14))
        width = rng.uniform(0.3, 0.6) * period
        d = (xx * np.cos(ang) + yy * np.sin(ang)) % period
        ink = (d < width).astype(np.float32)
        if style == "hatch2":  # cuadriculado (rayado cruzado)
            d2 = (xx * np.cos(ang + np.pi / 2) + yy * np.sin(ang + np.pi / 2)) % period
            ink = np.maximum(ink, (d2 < width).astype(np.float32))
        return ink
    if style == "dots":
        period = max(3.0, sq / rng.uniform(5, 10))
        r = rng.uniform(0.2, 0.4) * period
        d = np.hypot((xx % period) - period / 2, (yy % period) - period / 2)
        return (d < r).astype(np.float32)
    if style == "gray":
        return np.full((sq, sq), rng.uniform(0.25, 0.6), np.float32)
    return np.zeros((sq, sq), np.float32)


def render_diagram(b, pieces, rng, sq: int, flipped: bool) -> tuple[np.ndarray, int]:
    """Diagrama a tinta (0 = papel, 1 = tinta) con marco; devuelve (imagen, margen hasta el área de juego)."""
    B = 8 * sq
    pattern = dark_pattern(sq, rng)
    one_pattern = rng.random() < 0.85  # el mismo rayado en todas las casillas oscuras (lo normal en un libro)
    ink = np.zeros((B, B), np.float32)
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                ink[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = pattern if one_pattern else dark_pattern(sq, rng)
    if rng.random() < 0.3:  # líneas finas entre casillas
        for k in range(9):
            ink[:, min(B - 1, k * sq)] = np.maximum(ink[:, min(B - 1, k * sq)], 0.8)
            ink[min(B - 1, k * sq), :] = np.maximum(ink[min(B - 1, k * sq), :], 0.8)
    scale = rng.uniform(0.82, 1.0)
    ps = max(8, int(sq * scale))
    knock = rng.random() < 0.85
    halo = int(rng.integers(1, 3))
    cache = {}
    for rk in range(8):
        for fl in range(8):
            p = b[rk][fl]
            if p is None:
                continue
            if p not in cache:
                ip = ink_piece(cv2.resize(pieces[p], (ps, ps), interpolation=cv2.INTER_AREA), rng)
                cache[p] = ip
            ip = cache[p]
            r, c = (7 - rk, fl) if not flipped else (rk, 7 - fl)
            off = (sq - ps) // 2
            y0, x0 = r * sq + off, c * sq + off
            a = ip[..., 1]
            if knock:  # el rayado se corta alrededor de la pieza
                k = cv2.dilate((a > 0.3).astype(np.uint8), np.ones((2 * halo + 1, 2 * halo + 1), np.uint8))
                roi = ink[y0:y0 + ps, x0:x0 + ps]
                roi[k > 0] = 0
            roi = ink[y0:y0 + ps, x0:x0 + ps]
            ink[y0:y0 + ps, x0:x0 + ps] = roi * (1 - a) + ip[..., 0] * a
    # Marco: simple grueso, doble (fino + grueso) o fino.
    style = rng.choice(["thick", "double", "thin"], p=[.4, .4, .2])
    gap = int(rng.integers(0, 3))
    t1 = max(1, int(sq * rng.uniform(0.04, 0.12))) if style != "thin" else 1
    m = gap + t1 + (int(rng.integers(2, 5)) + 1 if style == "double" else 0) + 2
    out = np.zeros((B + 2 * m, B + 2 * m), np.float32)
    out[m:m + B, m:m + B] = ink
    cv2.rectangle(out, (m - gap - 1, m - gap - 1), (m + B + gap, m + B + gap), 1.0, t1)
    if style == "double":
        o = gap + t1 + int(rng.integers(2, 5))
        cv2.rectangle(out, (m - o - 1, m - o - 1), (m + B + o, m + B + o), 1.0, 1)
    return out, m


def page(diag: np.ndarray, m: int, sq: int, pieces, rng, flipped: bool) -> tuple[np.ndarray, np.ndarray]:
    """Página a tinta con el diagrama, coordenadas y texto alrededor; devuelve (tinta, esquinas)."""
    D = diag.shape[0]
    W = int(D * rng.uniform(1.6, 2.6))
    H = int(D * rng.uniform(1.5, 2.4))
    ink = np.zeros((H, W), np.float32)
    x0 = int(rng.integers(int(0.05 * W), max(int(0.05 * W) + 1, W - D - int(0.05 * W))))
    y0 = int(rng.integers(int(0.08 * H), max(int(0.08 * H) + 1, H - D - int(0.12 * H))))
    ink[y0:y0 + D, x0:x0 + D] = np.maximum(ink[y0:y0 + D, x0:x0 + D], diag)
    font = FONTS[int(rng.integers(len(FONTS)))]
    fs = sq / 55 * rng.uniform(0.8, 1.2)
    if rng.random() < 0.4:  # coordenadas fuera del marco
        files, ranks = ("abcdefgh", "87654321") if not flipped else ("hgfedcba", "12345678")
        for i in range(8):
            cv2.putText(ink, files[i], (x0 + m + i * sq + sq // 3, y0 + D + int(sq * 0.45)), font, fs * 0.8, 1.0, 1, cv2.LINE_AA)
            cv2.putText(ink, ranks[i], (x0 - int(sq * 0.4), y0 + m + i * sq + int(sq * 0.65)), font, fs * 0.8, 1.0, 1, cv2.LINE_AA)
    # Texto con notación figurine: líneas de «jugadas» con íconos de piezas chicos.
    lh = max(8, int(sq * rng.uniform(0.45, 0.75)))
    icons = [ink_piece(cv2.resize(pieces[p], (lh, lh), interpolation=cv2.INTER_AREA), rng) for p in ("wN", "wB", "wR", "wQ", "wK")]
    y = int(rng.integers(lh, 2 * lh))
    while y < H - 4:
        x = int(rng.integers(2, lh))
        while x < W - 3 * lh:
            if y0 - lh < y < y0 + D + lh and x0 - lh * 3 < x < x0 + D + lh:
                x = x0 + D + lh
                continue
            if rng.random() < 0.25:
                ic = icons[int(rng.integers(len(icons)))]
                hh, ww = ic.shape[:2]
                yy0, xx0 = max(0, y - hh), x
                if yy0 + hh <= H and xx0 + ww <= W:
                    roi = ink[yy0:yy0 + hh, xx0:xx0 + ww]
                    ink[yy0:yy0 + hh, xx0:xx0 + ww] = np.maximum(roi, ic[..., 0] * ic[..., 1])
                x += lh
            word = "".join(rng.choice(list("abcdefghxO-+12345678.,;!?enlasdpormqu"), size=int(rng.integers(2, 8))))
            (tw, _), _ = cv2.getTextSize(word, font, fs, 1)
            if x + tw < W:
                cv2.putText(ink, word, (x, y), font, fs, 1.0, 1, cv2.LINE_AA)
            x += tw + int(lh * 0.5)
        y += int(lh * rng.uniform(1.3, 1.7))
    corners = np.array([[x0 + m, y0 + m], [x0 + m + 8 * sq, y0 + m], [x0 + m + 8 * sq, y0 + m + 8 * sq],
                        [x0 + m, y0 + m + 8 * sq]], np.float64)
    return ink, corners


def print_effects(ink: np.ndarray, rng) -> np.ndarray:
    """Tinta → imagen RGB de papel impreso (color de papel, textura, tinta corrida o gastada)."""
    if rng.random() < 0.4:  # tinta corrida / impresión gorda
        ink = cv2.dilate(ink, np.ones((2, 2), np.uint8)) if rng.random() < 0.5 else cv2.GaussianBlur(ink, (0, 0), 0.7)
    if rng.random() < 0.3:  # impresión gastada: huecos en la tinta
        holes = cv2.GaussianBlur(rng.random(ink.shape).astype(np.float32), (0, 0), 1.2)
        ink = ink * (holes > rng.uniform(0.35, 0.48))
    paper = np.array([rng.uniform(225, 255), rng.uniform(215, 250), rng.uniform(185, 245)], np.float32)
    paper = np.minimum(paper, paper[0])  # el amarilleo baja el azul, no sube el rojo
    tex = cv2.GaussianBlur(rng.normal(0, 1, ink.shape).astype(np.float32), (0, 0), rng.uniform(1, 4)) * rng.uniform(2, 8)
    ink_col = rng.uniform(10, 60)
    img = paper[None, None, :] * (1 - ink[..., None]) + ink_col * ink[..., None] + tex[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def curvature(img: np.ndarray, corners: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Curvatura de página cerca del lomo (deformación suave, no proyectiva)."""
    H, W = img.shape[:2]
    side = rng.choice([-1, 1])  # lomo a la izquierda o a la derecha
    A = rng.uniform(0.01, 0.06) * H
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = xx / W if side > 0 else 1 - xx / W
    bend = A * u ** 3  # crece hacia el lomo
    comp = rng.uniform(0.0, 0.05) * W * u ** 3 * side
    mapx = (xx + comp).astype(np.float32)
    mapy = (yy + bend * (yy - H / 2) / (H / 2)).astype(np.float32)
    out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Posición de cada esquina en la imagen deformada: el píxel destino cuyo origen está más cerca.
    new = []
    for cx, cy in corners:
        d = (mapx - cx) ** 2 + (mapy - cy) ** 2
        j = int(np.argmin(d))
        new.append([j % W, j // W])
    return out, np.array(new, np.float64)


def photograph(img: np.ndarray, corners: np.ndarray, rng, scan: bool) -> tuple[np.ndarray, np.ndarray]:
    out_w, out_h = (960, 1280) if rng.random() < 0.6 else (1280, 960)
    D = np.linalg.norm(corners[1] - corners[0])
    size = rng.uniform(0.3, 0.8) * min(out_w, out_h) if not scan else rng.uniform(0.3, 0.6) * min(out_w, out_h)
    s = size / D
    H0 = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)
    # Colocar el tablero centrado aprox. y aplicar inclinación/giro alrededor de su centro.
    c = corners.mean(0) * s
    cx, cy = rng.uniform(0.35, 0.65) * out_w, rng.uniform(0.35, 0.65) * out_h
    if scan:
        ang = np.deg2rad(rng.uniform(-3, 3))
        tilt = 0.0
    else:
        ang = np.deg2rad(rng.uniform(-15, 15))
        tilt = rng.uniform(0, 0.3)
    src = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 100
    dst = src.copy()
    dst[:2, 0] *= 1 - tilt
    dst[:, 1] *= 1 - 0.5 * tilt
    if not scan:
        dst += rng.normal(0, 3, dst.shape).astype(np.float32)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = dst @ R.T.astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    T1 = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], np.float64)
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
    M = T2 @ P @ T1 @ H0
    bg = tuple(int(v) for v in rng.integers(20, 120, 3))
    photo = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    pc = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(4, 2)
    x = photo.astype(np.float32)
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    if scan:
        x = np.clip((x - 128) * rng.uniform(1.0, 1.4) + 128 + rng.uniform(-10, 20), 0, 255)
        if rng.random() < 0.5:
            x[:] = x.mean(2, keepdims=True)
    else:
        a = rng.uniform(0, 2 * np.pi)
        ramp = np.cos(a) * xx / out_w + np.sin(a) * yy / out_h
        x *= (1 + rng.uniform(-0.35, 0.35) * (ramp - ramp.mean()))[..., None]
        x *= rng.uniform(0.6, 1.15) * rng.uniform(0.88, 1.12, 3)
        if rng.random() < 0.35:  # sombra de la mano o del celular: borde suave desde un costado
            ang2 = rng.uniform(0, 2 * np.pi)
            d = (xx - out_w / 2) * np.cos(ang2) + (yy - out_h / 2) * np.sin(ang2)
            edge = rng.uniform(0.1, 0.45) * min(out_w, out_h)
            sh = 1 / (1 + np.exp(-(d - edge) / rng.uniform(15, 60)))
            x *= (1 - rng.uniform(0.3, 0.6) * sh)[..., None]
    photo = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.4, 1.6))
    if rng.random() < 0.6:
        photo = np.clip(photo + rng.normal(0, rng.uniform(1, 7), photo.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), pc


def sample(sets, rng):
    name = list(sets)[int(rng.integers(len(sets)))]
    b = random_board(rng)
    flipped = bool(rng.random() < 0.1)
    sq = int(rng.integers(18, 61))
    diag, m = render_diagram(b, sets[name], rng, sq, flipped)
    ink, corners = page(diag, m, sq, sets[name], rng, flipped)
    img = print_effects(ink, rng)
    scan = bool(rng.random() < 0.3)
    if not scan and rng.random() < 0.6:
        img, corners = curvature(img, corners, rng)
    photo, pc = photograph(img, corners, rng, scan)
    fen = to_fen(b)
    return photo, pc, g.board_labels(fen, "TR" if flipped else "BL"), fen, name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", type=Path, default=Path("data/pieces_png"))
    ap.add_argument("--out", type=Path, default=Path("data/preparedbook"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", nargs=4, type=int, default=[4000, 200, 300, 300], metavar=("TRAIN", "VAL", "TEST", "TEST_HELDOUT"))
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--preview", type=Path)
    ap.add_argument("--testdata", type=Path, help="escribir N fotos de prueba commiteables (solo set chessnut, Apache 2.0)")
    ap.add_argument("--count", type=int, default=12)
    a = ap.parse_args()
    if a.testdata:
        sets = {k: v for k, v in load_book_sets(a.pieces, "heldout").items() if k == "chessnut"}
        rng = np.random.default_rng([a.seed, 99])
        a.testdata.mkdir(parents=True, exist_ok=True)
        for i in range(a.count):
            photo, pc, lab, fen, _ = sample(sets, rng)
            cv2.imwrite(str(a.testdata / f"book_{i:02d}.jpg"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            (a.testdata / f"book_{i:02d}.json").write_text(json.dumps({
                "fen": fen, "a1_corner": "BL" if lab[56] == lab[56] and g.board_labels(fen, "BL").tolist() == lab.tolist() else "TR",
                "corners_TL_TR_BR_BL": pc.round(2).tolist(), "image_size_wh": [photo.shape[1], photo.shape[0]]}, indent=1))
        return
    seen, held = load_book_sets(a.pieces, "train"), load_book_sets(a.pieces, "heldout")
    for k, (split, n) in enumerate(zip(["train", "val", "test", "test_heldout"], a.n)):
        if (a.splits and split not in a.splits) or n == 0:
            continue
        sets = held if split == "test_heldout" else seen
        rng = np.random.default_rng([a.seed, k])
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        canon = np.lib.format.open_memmap(d / "canon.npy", mode="w+", dtype=np.uint8, shape=(n, g.CH, g.CW, 3))
        labels, names, corners_all = np.zeros((n, 64), np.int64), [], []
        for i in tqdm(range(n), desc=split):
            photo, pc, lab, fen, setname = sample(sets, rng)
            canon[i] = g.warp(photo, g.homography(pc))  # orden natural TL,TR,BR,BL (giro ≤ 15°)
            labels[i] = lab
            names.append(f"{setname} {fen}")
            if a.preview and i < 8 and split in ("val", "test_heldout"):
                a.preview.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(a.preview / f"{split}_photo{i}.jpg"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(a.preview / f"{split}_canon{i}.png"), cv2.cvtColor(canon[i], cv2.COLOR_RGB2BGR))
                (a.preview / f"{split}_photo{i}.json").write_text(
                    json.dumps({"fen": fen, "corners_TL_TR_BR_BL": pc.round(2).tolist()}))
        canon.flush()
        np.save(d / "labels.npy", labels)
        (d / "names.txt").write_text("\n".join(names) + "\n")


if __name__ == "__main__":
    main()
