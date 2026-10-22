"""Aumentación sobre la imagen canónica, antes del recorte.

La idea es imitar lo que pasa en el teléfono: esquinas detectadas con algunos
píxeles de error, menos resolución, desenfoque, ruido de sensor, compresión y
luz/color muy distintos a los del render.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import geometry as g


def jitter_geometry(canon: np.ndarray, rng: np.random.Generator, max_px: float) -> np.ndarray:
    """Re-warpea la canónica moviendo cada esquina del tablero hasta `max_px` px."""
    src = g.DST_CORNERS
    dst = (src + rng.uniform(-max_px, max_px, size=(4, 2))).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(canon, H, (g.CW, g.CH), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    # Ganancia por canal (balance de blancos), brillo, contraste.
    x *= rng.uniform(0.8, 1.2, size=3) * rng.uniform(0.6, 1.4)
    m = x.mean()
    x = (x - m) * rng.uniform(0.7, 1.3) + m
    # Saturación.
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.5, 1.4)
    # Gradiente de iluminación (sombra suave / spot).
    if rng.random() < 0.5:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, 2 * np.pi)
        ramp = (np.cos(a) * xx / w + np.sin(a) * yy / h)
        x *= (1 + rng.uniform(-0.4, 0.4) * (ramp - ramp.mean()))[..., None]
    x = np.clip(x, 0, 255)
    # Gamma.
    x = 255 * (x / 255) ** rng.uniform(0.7, 1.4)
    return x.astype(np.uint8)


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    if rng.random() < 0.7:  # menos resolución (tablero chico en el cuadro)
        f = rng.uniform(0.35, 1.0)
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:  # desenfoque / movimiento
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
        else:
            k = int(rng.integers(3, 8))
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1 / k
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
            ker = cv2.warpAffine(ker, M, (k, k))
            img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    if rng.random() < 0.5:  # ruido de sensor
        n = rng.normal(0, rng.uniform(2, 10), img.shape)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:  # JPEG
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 95))])
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


def augment_board(canon: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                  geo_px: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (canónica aumentada, etiquetas [64] ajustadas)."""
    if rng.random() < 0.5:  # espejo del tablero entero: col -> 7-col
        canon = canon[:, ::-1].copy()
        labels = labels.reshape(8, 8)[:, ::-1].reshape(64)
    canon = jitter_geometry(canon, rng, geo_px)
    canon = photometric(canon, rng)
    canon = degrade(canon, rng)
    return canon, labels
