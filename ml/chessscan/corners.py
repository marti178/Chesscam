"""Refinamiento de esquinas: ajusta la retícula 9×9 a los bordes de las casillas.

Parte de 4 esquinas aproximadas (a ojo, clic, o seguimiento) y las mueve para
maximizar el gradiente de la imagen sobre las 18 líneas de la retícula, en la
dirección perpendicular a cada línea. Lo usan la anotación de fotos reales y,
opcionalmente, el autoetiquetado. No es el detector de la app.
"""
from __future__ import annotations

import cv2
import numpy as np

_R = 40  # px por casilla en la vista de trabajo
_N = 8 * _R


def _grad_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)


def _score(gx, gy, corners: np.ndarray, inner_only: bool) -> float:
    dst = np.float32([[0, 0], [_N, 0], [_N, _N], [0, _N]])
    H = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))  # vista -> imagen
    t = np.linspace(_R * 0.15, _N - _R * 0.15, 200)  # evita los cruces exactos
    ks = range(1, 8) if inner_only else range(9)
    total = 0.0
    for k in ks:
        for vertical in (True, False):
            pts = np.stack([np.full_like(t, k * _R), t], 1) if vertical else np.stack([t, np.full_like(t, k * _R)], 1)
            p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            # Normal de la línea en la imagen.
            q = pts + ([1, 0] if vertical else [0, 1])
            q = cv2.perspectiveTransform(q.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            n = q - p
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            x = np.clip(p[:, 0], 0, gx.shape[1] - 1).astype(np.float32)
            y = np.clip(p[:, 1], 0, gx.shape[0] - 1).astype(np.float32)
            vx = cv2.remap(gx, x.reshape(1, -1), y.reshape(1, -1), cv2.INTER_LINEAR).ravel()
            vy = cv2.remap(gy, x.reshape(1, -1), y.reshape(1, -1), cv2.INTER_LINEAR).ravel()
            # En las líneas internas el contraste cambia de signo casilla a casilla: usar |.|
            total += np.abs(vx * n[:, 0] + vy * n[:, 1]).mean()
    return total


def refine(img_rgb: np.ndarray, corners: np.ndarray, radius: float = 12.0, iters: int = 6,
           inner_only: bool = False) -> np.ndarray:
    """Búsqueda por coordenadas con paso decreciente sobre las 8 coordenadas."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gx, gy = _grad_maps(gray)
    c = np.asarray(corners, np.float64).reshape(4, 2).copy()
    best = _score(gx, gy, c, inner_only)
    step = radius / 2
    for _ in range(iters):
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for j in range(2):
                    for s in (-step, step):
                        cand = c.copy()
                        cand[i, j] += s
                        v = _score(gx, gy, cand, inner_only)
                        if v > best:
                            best, c, improved = v, cand, True
        step /= 2
    return c
