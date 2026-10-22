"""Rasteriza los sets de piezas SVG de Lichess (lila/public/piece) a PNG RGBA.

Uso:  python -m chessscan.pieces2d --svg data/raw/pieces --out data/pieces_png
Licencia de cada set: ml/LICENSES.md.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import resvg_py

PIECES = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"]
# Sets reservados para el test 2D: nunca se usan para entrenar.
TEST_SETS = {"chessnut", "staunty"}


def rasterize(svg: Path, size: int) -> np.ndarray:
    text = svg.read_text(encoding="utf-8")
    try:
        png = resvg_py.svg_to_bytes(svg_string=text, width=size, height=size)
    except ValueError:
        # Tamaños raros (sin tamaño, o en mm): se reemplazan por los del viewBox.
        head_end = text.index(">", text.index("<svg"))
        head = re.sub(r'\s(width|height)="[^"]*"', "", text[:head_end])
        vb = re.search(r'viewBox="([\d.\s-]+)"', head)
        w, h = (vb.group(1).split()[2:] if vb else ("45", "45"))
        text = head.replace("<svg", f'<svg width="{w}" height="{h}"', 1) + text[head_end:]
        png = resvg_py.svg_to_bytes(svg_string=text, width=size, height=size)
    im = cv2.imdecode(np.frombuffer(bytes(png), np.uint8), cv2.IMREAD_UNCHANGED)
    if im.shape[0] != size or im.shape[1] != size:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return im  # BGRA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=160)
    a = ap.parse_args()
    for s in sorted(p for p in a.svg.iterdir() if p.is_dir()):
        (a.out / s.name).mkdir(parents=True, exist_ok=True)
        for p in PIECES:
            cv2.imwrite(str(a.out / s.name / f"{p}.png"), rasterize(s / f"{p}.svg", a.size))
        print(s.name)


if __name__ == "__main__":
    main()
