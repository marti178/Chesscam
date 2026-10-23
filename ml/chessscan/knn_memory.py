"""Simulación offline de la «memoria de ejemplos» (correcciones del usuario + kNN sobre embeddings).

Videos físicos que nunca se usaron para entrenar (132356 y 133425). Correcciones: unos pocos
cuadros del primer 20 % del video; test: 1 de cada 5 cuadros del último 60 % (separados en el tiempo).
Dos formas de memoria: «todas» (el usuario confirma el tablero entero de esos cuadros) y «errores»
(solo guarda las casillas que el modelo erró). Mezcla: p = (1 − w)·softmax(logits) + w·voto kNN, donde
el voto usa los k vecinos con similitud coseno ≥ umbral, ponderados por similitud; sin vecinos, queda el modelo.
Los parámetros se eligen en un video y se prueban en el otro.

  python -m chessscan.knn_memory --model export/chess_square_classifier_emb.tflite --out runs/knn_small.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import geometry as g
from .data import Boards

VIDEOS = {"132356": "data/real/video4", "133425": "data/real/video5"}


def run_model(path: Path, canon, idx, threads: int):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    od = it.get_output_details()
    i_lo = next(d["index"] for d in od if d["shape"][-1] == len(g.CLASSES))
    i_em = next(d["index"] for d in od if d["shape"][-1] != len(g.CLASSES))
    lo, em = [], []
    for k in idx:
        it.set_tensor(it.get_input_details()[0]["index"], g.model_input(g.crop_all(np.asarray(canon[k]))))
        it.invoke()
        lo.append(it.get_tensor(i_lo).copy())
        em.append(it.get_tensor(i_em).copy())
    return np.stack(lo), np.stack(em)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def combine(p_model, emb, mem_e, mem_y, k, thr, w):
    if len(mem_e) == 0:
        return p_model
    sim = emb @ mem_e.T
    k = min(k, len(mem_e))
    nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(sim, nn, 1)
    vote = np.zeros_like(p_model)
    ok = s >= thr
    for j in range(k):
        np.add.at(vote, (np.arange(len(vote))[ok[:, j]], mem_y[nn[ok[:, j], j]]), s[ok[:, j], j])
    has = vote.sum(1) > 0
    vote[has] /= vote[has].sum(1, keepdims=True)
    out = p_model.copy()
    out[has] = (1 - w) * p_model[has] + w * vote[has]
    return out


def prepare(model, path, n_corr, threads):
    b = Boards.load(Path(path))
    n = len(b)
    corr = np.linspace(0, int(0.2 * n) - 1, n_corr).astype(int)
    test = np.arange(int(0.4 * n), n, 5)
    lc, ec = run_model(model, b.canon, corr, threads)
    lt, et = run_model(model, b.canon, test, threads)
    return dict(lc=lc.reshape(-1, 13), ec=ec.reshape(len(corr) * 64, -1), yc=b.labels[corr].reshape(-1),
                lt=lt.reshape(-1, 13), et=et.reshape(len(test) * 64, -1), yt=b.labels[test].reshape(-1))


def score(d, mode, k, thr, w):
    keep = d["yc"] >= 0
    if mode == "errores":
        keep &= d["lc"].argmax(1) != d["yc"]
    me, my = d["ec"][keep], d["yc"][keep]
    v = d["yt"] >= 0
    pm = softmax(d["lt"][v])
    y = d["yt"][v]
    occ = y != 0
    res = {}
    for name, ww in (("modelo", None), ("knn", 1.0), ("mezcla", w)):
        p = pm if ww is None else combine(pm, d["et"][v], me, my, k, thr, ww)
        pr = p.argmax(1)
        res[name] = {"casillas": float((pr == y).mean()), "piezas": float((pr[occ] == y[occ]).mean())}
    res["memoria"] = int(keep.sum())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-corr", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    grid = list(itertools.product((1, 3, 5, 10), (0.5, 0.6, 0.7, 0.8, 0.9), (0.3, 0.5, 0.7)))
    out = {}
    for nc in a.n_corr:
        data = {v: prepare(a.model, p, nc, a.threads) for v, p in VIDEOS.items()}
        for mode in ("todas", "errores"):
            for sel, other in (("132356", "133425"), ("133425", "132356")):
                best = max(grid, key=lambda q: score(data[sel], mode, *q)["mezcla"]["casillas"])
                r = {"params_k_umbral_w": best, "elegido_en": sel,
                     "en_" + sel: score(data[sel], mode, *best), "en_" + other: score(data[other], mode, *best)}
                out[f"{nc}_cuadros/{mode}/elegido_en_{sel}"] = r
                print(nc, mode, json.dumps(r), flush=True)
            for q in ((5, 0.7, 0.5), (3, 0.8, 0.5), (10, 0.7, 0.5)):
                out[f"{nc}_cuadros/{mode}/fijo_{q}"] = {v: score(data[v], mode, *q) for v in VIDEOS}
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
