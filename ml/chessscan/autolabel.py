"""Autoetiquetado: video de un tablero real en una posición conocida → recortes etiquetados.

Como la posición (FEN) no cambia durante el video, las 64 casillas de cada
fotograma se etiquetan solas. Solo hace falta ubicar el tablero en cada fotograma:

1. En el primer fotograma se marcan las 4 esquinas exteriores del área de juego
   (clic con el mouse, o `--corners` si ya se conocen) en el orden TL, TR, BR, BL
   de CONTRACT §1 (TL→TR = borde lejano a la cámara).
2. En los siguientes se siguen con SIFT + homografía RANSAC contra un fotograma
   clave (el plano del tablero domina; las piezas quedan como outliers).
   Si se camina alrededor del tablero, en cada fotograma se reordenan las
   esquinas para que TL→TR siga siendo el borde de arriba (lejano) y la
   cuadrícula de etiquetas se rota igual.
3. Se descartan fotogramas movidos, con pocas correspondencias o con el tablero
   fuera de cuadro, y se guarda uno cada `--every`.

Salida (mismo formato que prepare_chesscog, se puede pasar a pipeline con --extra):
  <out>/canon.npy, labels.npy, names.txt, corners.json, preview.jpg

Ejemplo (posición inicial, blancas del lado de la cámara):
  python -m chessscan.autolabel video.mp4 --out data/real/mesa_1 \
      --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" --a1 BL

`--a1` dice en qué esquina de la retícula está a1: BL = blancas cerca de la
cámara, TR = blancas del lado lejano, TL = blancas a la izquierda, BR = a la derecha.
Revisar siempre preview.jpg antes de usar los datos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry as g


def click_corners(frame_rgb: np.ndarray) -> np.ndarray:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.imshow(frame_rgb)
    plt.title("Clic en las 4 esquinas del área de juego: lejana-izq, lejana-der, cercana-der, cercana-izq")
    pts = plt.ginput(4, timeout=0)
    plt.close()
    if len(pts) != 4:
        raise SystemExit("hacían falta 4 clics")
    return np.array(pts, dtype=np.float64)


def board_mask(shape, corners: np.ndarray, grow: float = 0.15) -> np.ndarray:
    c = corners.mean(0)
    poly = (c + (corners - c) * (1 + grow)).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, poly, 255)
    return m


class Tracker:
    def __init__(self, gray: np.ndarray, corners: np.ndarray, min_inliers: int = 60):
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_inliers = min_inliers
        self._set_key(gray, corners)

    def _set_key(self, gray, corners):
        self.key_corners = corners.copy()
        self.key_kp, self.key_des = self.sift.detectAndCompute(gray, board_mask(gray.shape, corners))

    def track(self, gray: np.ndarray) -> tuple[np.ndarray | None, int]:
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or self.key_des is None or len(kp) < 10:
            return None, 0
        good = []
        for m in self.matcher.knnMatch(self.key_des, des, k=2):
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good.append(m[0])
        if len(good) < self.min_inliers:
            return None, len(good)
        src = np.float32([self.key_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        if H is None or n < self.min_inliers:
            return None, n
        corners = cv2.perspectiveTransform(self.key_corners.reshape(-1, 1, 2), H).reshape(4, 2)
        # Si la vista ya cambió bastante, este fotograma pasa a ser el nuevo clave.
        if n < 3 * self.min_inliers:
            self._set_key(gray, corners)
        return corners, n


def corners_ok(corners: np.ndarray, shape, min_side: float = 120) -> bool:
    h, w = shape[:2]
    if (corners < -2).any() or (corners[:, 0] > w + 2).any() or (corners[:, 1] > h + 2).any():
        return False
    sides = np.linalg.norm(corners - np.roll(corners, -1, 0), axis=1)
    if sides.min() < min_side:
        return False
    # Convexo y en el orden de CONTRACT (TL→TR→BR→BL = horario en coordenadas de imagen).
    v = np.roll(corners, -1, 0) - corners
    cross = v[:, 0] * np.roll(v, -1, 0)[:, 1] - v[:, 1] * np.roll(v, -1, 0)[:, 0]
    return bool((cross > 0).all())


def upright(corners: np.ndarray, labels0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rota el orden de las esquinas seguidas para que queden TL, TR, BR, BL en esta
    imagen, y rota las etiquetas del primer fotograma en consecuencia."""
    k = int(np.argmin(np.linalg.norm(corners - g.sort_corners(corners)[0], axis=1)))
    return np.roll(corners, -k, axis=0), np.rot90(labels0.reshape(8, 8), k).reshape(64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fen", required=True)
    ap.add_argument("--a1", default="BL", choices=["BL", "TR", "TL", "BR"])
    ap.add_argument("--corners", help="'x,y x,y x,y x,y' en el primer fotograma (TL TR BR BL)")
    ap.add_argument("--every", type=int, default=5, help="guardar 1 de cada N fotogramas buenos")
    ap.add_argument("--max-side", type=int, default=1280, help="reducir fotogramas a este lado máximo")
    ap.add_argument("--min-sharpness", type=float, default=60.0, help="varianza del Laplaciano mínima")
    ap.add_argument("--max-frames", type=int, default=400)
    a = ap.parse_args()

    labels = g.board_labels(a.fen, a.a1)
    cap = cv2.VideoCapture(str(a.video))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"no se pudo leer {a.video}")

    def prep(f):
        s = a.max_side / max(f.shape[:2])
        if s < 1:
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    rgb = prep(frame)
    if a.corners:
        corners = np.array([[float(v) for v in p.split(",")] for p in a.corners.split()])
        s = a.max_side / max(frame.shape[:2])
        corners = corners * (s if s < 1 else 1)
    else:
        corners = click_corners(rgb)
    if not corners_ok(corners, rgb.shape):
        raise SystemExit("las esquinas no forman un cuadrilátero válido en orden TL, TR, BR, BL")

    tracker = Tracker(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), corners)
    canons, names, all_corners = [g.warp(rgb, g.homography(corners))], ["f000000"], [corners.tolist()]
    all_labels = [labels]
    idx, good, stats = 0, 0, {"leidos": 1, "sin_seguimiento": 0, "fuera_de_cuadro": 0, "movidos": 0}
    while len(canons) < a.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        stats["leidos"] += 1
        rgb = prep(frame)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        c, _ = tracker.track(gray)
        if c is None:
            stats["sin_seguimiento"] += 1
            continue
        c, lab = upright(c, labels)
        if not corners_ok(c, rgb.shape):
            stats["fuera_de_cuadro"] += 1
            continue
        canon = g.warp(rgb, g.homography(c))
        sharp = cv2.Laplacian(cv2.cvtColor(canon, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        if sharp < a.min_sharpness:
            stats["movidos"] += 1
            continue
        good += 1
        if good % a.every:
            continue
        canons.append(canon)
        names.append(f"f{idx:06d}")
        all_corners.append(c.tolist())
        all_labels.append(lab)

    a.out.mkdir(parents=True, exist_ok=True)
    canon_arr = np.stack(canons)
    np.save(a.out / "canon.npy", canon_arr)
    label_arr = np.stack(all_labels)
    np.save(a.out / "labels.npy", label_arr)
    (a.out / "names.txt").write_text("\n".join(names) + "\n")
    (a.out / "corners.json").write_text(json.dumps({
        "video": str(a.video), "fen": a.fen, "a1": a.a1, "frames": dict(zip(names, all_corners)),
        "stats": stats}, indent=1))

    # Vista previa: hasta 6 canónicas con la retícula y las etiquetas, para revisar a ojo.
    pick = np.linspace(0, len(canons) - 1, min(6, len(canons))).astype(int)
    tiles = []
    for i in pick:
        t = cv2.cvtColor(canon_arr[i], cv2.COLOR_RGB2BGR).copy()
        for k in range(9):
            cv2.line(t, (g.MX + g.S * k, g.MY), (g.MX + g.S * k, g.CH - 1), (0, 255, 0), 1)
            cv2.line(t, (g.MX, g.MY + g.S * k), (g.MX + 8 * g.S, g.MY + g.S * k), (0, 255, 0), 1)
        for r in range(8):
            for col in range(8):
                lab = g.CLASSES[label_arr[i][r * 8 + col]]
                if lab != "empty":
                    cv2.putText(t, lab, (g.MX + g.S * col + 10, g.MY + g.S * r + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(a.out / "preview.jpg"), np.vstack(rows))
    print(f"{len(canons)} fotogramas guardados en {a.out}; {stats}")


if __name__ == "__main__":
    main()
