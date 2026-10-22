"""Implementación de referencia de ml/CONTRACT.md (v1).

Rectificación del tablero y recorte de las 64 casillas. Todo lo que hace este
módulo tiene que poder reproducirse en Kotlin con OpenCV de forma idéntica.
"""
from __future__ import annotations

import cv2
import numpy as np

CONTRACT_VERSION = 1

S = 32                      # lado de una casilla en la imagen canónica
MX = S                      # margen lateral
MY = 3 * S                  # margen superior
CW = 8 * S + 2 * MX         # 320
CH = 8 * S + MY             # 352
CROP_H = 4 * S              # 128
CROP_W = 2 * S              # 64

CLASSES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}

DST_CORNERS = np.array(
    [[MX, MY], [MX + 8 * S, MY], [MX + 8 * S, MY + 8 * S], [MX, MY + 8 * S]],
    dtype=np.float32,
)


def homography(corners: np.ndarray) -> np.ndarray:
    """H (3x3, float64) imagen -> canónica. `corners` = [TL, TR, BR, BL] en píxeles."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, DST_CORNERS)


def warp(img_rgb: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Imagen canónica RGB uint8 de CH x CW."""
    return cv2.warpPerspective(
        img_rgb, H, (CW, CH), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )


def row_bounds(row: int) -> tuple[int, int]:
    up = S + (2 * S * (7 - row)) // 7
    return MY + S * row - up, MY + S * (row + 1)


def col_bounds(col: int) -> tuple[int, int]:
    d = 3 - col if col < 4 else col - 4
    extra = (S // 4) * (d + 1)
    if col < 4:
        return MX + S * col - extra, MX + S * (col + 1)
    return MX + S * col, MX + S * (col + 1) + extra


def crop_square(canon: np.ndarray, row: int, col: int) -> np.ndarray:
    """Recorte CROP_H x CROP_W x 3 de la casilla (fila, col)."""
    y0, y1 = row_bounds(row)
    x0, x1 = col_bounds(col)
    patch = canon[y0:y1, x0:x1]
    if col < 4:
        patch = cv2.flip(patch, 1)
    out = np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)
    out[CROP_H - (y1 - y0):, : x1 - x0] = patch
    return out


def crop_all(canon: np.ndarray) -> np.ndarray:
    """Lote uint8 [64, CROP_H, CROP_W, 3], índice = fila*8 + col."""
    return np.stack([crop_square(canon, r, c) for r in range(8) for c in range(8)])


def crops_from_image(img_rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return crop_all(warp(img_rgb, homography(corners)))


def model_input(crops: np.ndarray) -> np.ndarray:
    """Tensor de entrada del modelo: float32 [64,128,64,3], valores 0..255."""
    return crops.astype(np.float32)


def sort_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como TL, TR, BR, BL según su posición en la imagen derecha.

    Supone que el borde superior de la imagen es el lejano (ver CONTRACT §1).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # sentido horario en coordenadas de imagen, empezando cerca de -pi
    # Rotar para que el primero sea el de menor x+y (arriba-izquierda).
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


# --- Correspondencia entre la retícula y las casillas de ajedrez -------------

def grid_to_square(row: int, col: int, a1: str = "BL") -> str:
    """Casilla algebraica de (fila, col) según en qué esquina de la retícula esté a1.

    a1 = "BL": blancas del lado cercano a la cámara; "TR": del lado lejano;
    "TL": blancas a la izquierda de la imagen; "BR": a la derecha.
    """
    if a1 == "BL":
        f, r = col, 7 - row
    elif a1 == "TR":
        f, r = 7 - col, row
    elif a1 == "TL":
        f, r = row, col
    elif a1 == "BR":
        f, r = 7 - row, 7 - col
    else:
        raise ValueError(f"esquina de a1 inválida: {a1}")
    return "abcdefgh"[f] + str(r + 1)


def board_labels(fen: str, a1: str = "BL") -> np.ndarray:
    """Etiquetas [64] (índices de CLASSES, orden fila*8+col) para un FEN."""
    import chess

    board = chess.Board(fen.split()[0] + " w - - 0 1")
    labels = np.zeros(64, dtype=np.int64)
    for row in range(8):
        for col in range(8):
            p = board.piece_at(chess.parse_square(grid_to_square(row, col, a1)))
            labels[row * 8 + col] = CLASS_TO_INDEX[p.symbol()] if p else 0
    return labels
