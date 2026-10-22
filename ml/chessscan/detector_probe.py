"""Reproduce la primera etapa del detector de Android (android/core/.../BoardDetector.kt) para diagnosticar.

Mismos parámetros que DetectorParams: imagen de trabajo de 640 px de lado mayor,
GaussianBlur 5×5, Canny(40, 120), HoughLines(1 px, π/360, umbral = 0,16 × lado menor),
120 líneas como máximo e histograma de ángulos con dos picos separados ≥ 35°.
Opcionalmente aplica un prefiltro antes (para probar arreglos) y mide cuántas de
las líneas detectadas coinciden con alguna de las 18 líneas reales de la retícula.

  python -m chessscan.detector_probe imagen.png --corners "x,y x,y x,y x,y" --prefilter none|open|blur --out probe.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def prefilter(gray: np.ndarray, mode: str, sq_px: float) -> np.ndarray:
    if mode == "none":
        return gray
    if mode == "open":  # apertura (mín → máx): los huecos claros entre rayas desaparecen y la casilla queda oscura lisa
        k = max(3, int(round(sq_px * 0.25)) | 1)
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if mode == "blur":  # promedio del tamaño del período del rayado
        k = max(3, int(round(sq_px * 0.35)) | 1)
        return cv2.blur(gray, (k, k))
    raise ValueError(mode)


def probe(img_rgb: np.ndarray, corners: np.ndarray | None, mode: str, sq_px_hint: float | None = None) -> dict:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    s = min(1.0, 640 / max(gray.shape))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray.copy()
    c = corners * s if corners is not None else None
    sq = sq_px_hint * s if sq_px_hint else (np.linalg.norm(c[1] - c[0]) / 8 if c is not None else 30)
    g2 = prefilter(g, mode, sq)
    blurred = cv2.GaussianBlur(g2, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    thr = max(30, int(round(0.16 * min(g.shape))))
    lines = cv2.HoughLines(edges, 1, np.pi / 360, thr)
    lines = [] if lines is None else [tuple(l[0]) for l in lines[:120]]
    out = {"work_size": g.shape[::-1], "edge_px": int((edges > 0).sum()), "hough_lines": len(lines),
           "prefilter": mode}
    deg = np.array([((np.rad2deg(t) % 180) + 180) % 180 for _, t in lines])
    if len(deg):
        hist = np.zeros(180)
        for i, d in enumerate(deg):
            w = 1 / (1 + i / 20)
            for k in range(-3, 4):
                hist[(int(d) + k) % 180] += w * (4 - abs(k))
        p1 = int(hist.argmax())
        far = [i for i in range(180) if min(abs(i - p1) % 180, 180 - abs(i - p1) % 180) >= 35]
        p2 = max(far, key=lambda i: hist[i])
        out["angle_peaks_deg"] = [p1, p2]
    if c is not None and len(lines):
        # Ángulos de la retícula real (normales de las líneas de cada familia).
        Hi = cv2.getPerspectiveTransform(np.float32([[0, 0], [8, 0], [8, 8], [0, 8]]), c.astype(np.float32))
        grid = []
        for k in range(9):
            for a_, b_ in (((k, 0), (k, 8)), ((0, k), (8, k))):
                p = cv2.perspectiveTransform(np.float32([[a_, b_]]), Hi)[0]
                d = p[1] - p[0]
                n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                grid.append((float(n @ p[0]), float(np.arctan2(n[1], n[0]))))
        def matches(rho, th):
            for gr, gt in grid:
                for sgn in (1, -1):
                    if abs(((th - gt + np.pi / 2) % np.pi) - np.pi / 2) < np.deg2rad(3) and abs(sgn * rho - gr) < 4:
                        return True
            return False
        good = [matches(r, t) for r, t in lines]
        out["lines_on_grid"] = int(sum(good))
        out["grid_lines_found"] = int(sum(any(abs(((t - gt + np.pi / 2) % np.pi) - np.pi / 2) < np.deg2rad(3) and
                                                  min(abs(r - gr), abs(-r - gr)) < 4 for r, t in lines) for gr, gt in grid))
        gdeg = sorted({int(round(((np.rad2deg(t) % 180) + 180) % 180)) for _, t in grid})
        out["grid_angles_deg"] = [gdeg[0], gdeg[-1]]
    out["_vis"] = (g, edges, lines)
    return out


def draw(res: dict, path: Path) -> None:
    g, edges, lines = res["_vis"]
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for rho, th in lines:
        a, b = np.cos(th), np.sin(th)
        x0, y0 = a * rho, b * rho
        cv2.line(vis, (int(x0 + 2000 * -b), int(y0 + 2000 * a)), (int(x0 - 2000 * -b), int(y0 - 2000 * a)), (0, 0, 255), 1)
    cv2.imwrite(str(path), np.hstack([vis, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--corners")
    ap.add_argument("--prefilter", default="none", choices=["none", "open", "blur"])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    img = cv2.cvtColor(cv2.imread(str(a.image)), cv2.COLOR_BGR2RGB)
    c = np.array([[float(v) for v in p.split(",")] for p in a.corners.split()]) if a.corners else None
    res = probe(img, c, a.prefilter)
    if a.out:
        draw(res, a.out)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}))


if __name__ == "__main__":
    main()
