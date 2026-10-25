"""Calibración de la probabilidad por casilla del detector de piezas, para que la fusión de cuadros sume.

Modelo (CONTRACT_DETECTOR.md §Fusión): si la casilla tiene detección de clase c y confianza p,
  q = sigmoid(a · logit(p) + b_c)       → P(c) = q, P(vacía) = (1 − q) · e, resto (1 − q)(1 − e)/11 cada una;
sin detección: P(vacía) = π0 y el resto (1 − π0)/12 cada una. Parámetros: a, b (uno para todas las clases), e, π0, ajustados
por máxima verosimilitud por cuadro. Se ajusta dejando afuera el video que se mide (validación cruzada).

  python -m chessscan.det_calib dump --model export/piece_detector.tflite
  python -m chessscan.det_calib fit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry as g
from .det_eval import VIDEOS, summarize

OUT = Path("runs/det_v1/calib")
TEST = ["132356", "133425", "tablero_nuevo_215505", "libro_monitor"]


def dump(a):
    import cv2
    from .det_eval import TFLiteDetector
    det = TFLiteDetector(a.model, 0.05, 0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in TEST:
        d, raw, rng_ = VIDEOS[name]
        d = Path(d)
        meta = json.loads((d / "frames.json").read_text())
        names = (d / "names.txt").read_text().split()
        labels = np.load(d / "labels.npy")
        frames = np.load(raw, mmap_mode="r")
        sel = [k for k, n in enumerate(names) if rng_ is None or rng_[0] <= int(n[1:]) <= rng_[1]]
        C = np.zeros((len(sel), 64), np.int64)
        P = np.zeros((len(sel), 64), np.float32)
        for j, k in enumerate(sel):
            i = int(names[k][1:])
            c = np.array(meta["frames"][str(i)]["corners"], float)
            H = g.homography(c)
            for cl, p, x0, y0, x1, y1 in det(np.asarray(frames[i]), c):
                up = 0.35 * min(x1 - x0, y1 - y0)
                q = cv2.perspectiveTransform(np.float32([[[(x0 + x1) / 2, y1 - up]]]), H)[0, 0]
                col, row = (q[0] - g.MX) / g.S, (q[1] - g.MY) / g.S
                if 0 <= col < 8 and 0 <= row < 8:
                    s = int(row) * 8 + int(col)
                    if p > P[j, s]:
                        C[j, s], P[j, s] = cl, p
        np.savez(OUT / f"{name}.npz", cls=C, p=P, labels=labels[sel], ref=g.board_labels(meta["fen"], meta["a1_anchor"]))
        print(name, len(sel), flush=True)


def logprob(C, P, th):
    """th = [a, e, π0, b_1..b_12] → log P [N, 64, 13]."""
    a, e, pi0, b = th[0], th[1], th[2], th[3:]
    out = np.empty(C.shape + (13,))
    out[:] = (1 - pi0) / 12
    out[..., 0] = pi0
    has = C > 0
    p = np.clip(P[has], 1e-4, 1 - 1e-4)
    q = 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b[C[has] - 1])))
    rows = np.repeat(((1 - q) * (1 - e) / 11)[:, None], 13, 1)
    rows[:, 0] = (1 - q) * e
    rows[np.arange(len(q)), C[has]] = q
    out[has] = rows
    return np.log(np.clip(out, 1e-6, 1))


def baseline(C, P, t=0.35):
    """La conversión de det_v1 sin calibrar (det_eval.assign)."""
    out = np.full(C.shape + (13,), 0.05 / 12)
    out[..., 0] = 0.95
    has = C > 0
    p = np.clip(P[has], 1e-4, 1 - 1e-4)
    q = 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(t / (1 - t)))))
    rows = np.repeat(((1 - q) * 0.05 / 11)[:, None], 13, 1)
    rows[:, 0] = (1 - q) * 0.95
    rows[np.arange(len(q)), C[has]] = q
    out[has] = rows
    return np.log(out)


def nll(th, data):
    tot = n = 0.0
    for C, P, L in data:
        lp = logprob(C, P, th)
        v = L >= 0
        tot -= np.take_along_axis(lp[v], L[v][:, None], 1).sum()
        n += v.sum()
    return tot / n


def fitp(data, shared: bool = True):
    """shared: un solo sesgo b para las 12 clases (no cambia el orden de las confianzas dentro de un cuadro,
    solo cuánto pesa cada cuadro en la fusión). Con sesgo por clase se sobreajusta a los videos de ajuste."""
    from scipy.optimize import minimize
    if shared:
        f = lambda x: nll(np.concatenate([x[:3], np.full(12, x[3])]), data)
        r = minimize(f, [1.2, 0.6, 0.95, 0.3], method="L-BFGS-B", bounds=[(0.05, 10), (0.01, 0.999), (0.5, 0.9999), (-6, 6)],
                     options={"maxiter": 300, "eps": 1e-4})
        return np.concatenate([r.x[:3], np.full(12, r.x[3])])
    x0 = np.concatenate([[1.0, 0.9, 0.95], np.zeros(12)])
    bounds = [(0.05, 10), (0.01, 0.999), (0.5, 0.9999)] + [(-6, 6)] * 12
    return minimize(nll, x0, args=(data,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 300, "eps": 1e-4}).x


def fit(a):
    sets = {n: np.load(OUT / f"{n}.npz") for n in TEST}
    rows, params = {}, {}
    # Parámetros finales: ajustados con los 3 videos físicos (el libro tiene otra distribución de confianzas).
    phys = ["132356", "133425", "tablero_nuevo_215505"]
    for name in TEST:
        train = [n for n in phys if n != name]
        th = fitp([(sets[n]["cls"], sets[n]["p"], sets[n]["labels"]) for n in train])
        params[name] = th
        s = sets[name]
        rows[name] = {"sin_calibrar": summarize(baseline(s["cls"], s["p"]), s["labels"], s["ref"], (1, 30, 60, 0)),
                      "calibrado": summarize(logprob(s["cls"], s["p"], th), s["labels"], s["ref"], (1, 30, 60, 0)),
                      "ajustado_con": train}
        for k in ("sin_calibrar", "calibrado"):
            r = rows[name][k]
            print(f"{name:22s} {k:13s} " + "  ".join(f"{w}: {r[w]['square_acc'] * 100:.1f}/{r[w]['occupied_acc'] * 100:.1f}"
                                                      for w in ("ventana_1", "ventana_30", "ventana_60", "todo_el_tramo")), flush=True)
    final = fitp([(sets[n]["cls"], sets[n]["p"], sets[n]["labels"]) for n in phys])
    names = ["a", "e", "pi0"] + [f"b_{c}" for c in g.CLASSES[1:]]
    out = {"parametros_finales": dict(zip(names, map(float, final))), "por_video": rows}
    (OUT / "calib.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["parametros_finales"], indent=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "fit"])
    ap.add_argument("--model", type=Path, default=Path("export/piece_detector.tflite"))
    a = ap.parse_args()
    {"dump": dump, "fit": fit}[a.cmd](a)


if __name__ == "__main__":
    main()
