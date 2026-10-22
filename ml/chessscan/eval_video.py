"""Evalúa un .tflite sobre un video autoetiquetado: por cuadro y con fusión temporal simulada.

La fusión suma, por casilla física del tablero (no de la retícula de la imagen),
la log-probabilidad de cada clase en una ventana de cuadros consecutivos, como
propone la SPEC (sin decaimiento ni pesos por nitidez: es la versión más simple).
La casilla física de cada celda se deduce rotando la retícula de etiquetas hasta
que coincide con la del primer cuadro.

  python -m chessscan.eval_video --model export/chess_square_classifier.tflite --data data/real/video1 \
      --frames 740 1135 --windows 1 30 60 0
(ventana 0 = todo el tramo fusionado en una sola estimación)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry as g
from .data import Boards
from .pipeline import tflite_predict


def rotation_to_ref(lab: np.ndarray, ref: np.ndarray) -> int:
    """k tal que rot90(ref, k) coincide con lab en las casillas visibles."""
    L, R = lab.reshape(8, 8), ref.reshape(8, 8)
    vis = L >= 0
    best = max(range(4), key=lambda k: ((np.rot90(R, k) == L) & vis).sum())
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fen", help="FEN (si no, se lee de frames.json)")
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 10 ** 9], help="tramo [desde, hasta] de cuadros del video")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 30, 60, 0])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    meta = json.loads((a.data / "frames.json").read_text())
    ref = g.board_labels(a.fen or meta["fen"], meta["a1_anchor"])  # retícula del cuadro ancla
    names = (a.data / "names.txt").read_text().split()
    idx = np.array([int(n[1:]) for n in names])
    sel = np.where((idx >= a.frames[0]) & (idx <= a.frames[1]))[0]
    b = Boards.load(a.data)
    labels = b.labels[sel]
    x = np.concatenate([g.crop_all(np.asarray(b.canon[i])) for i in sel])
    logits = tflite_predict(a.model, x).reshape(len(sel), 64, -1)
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    # Pasar todo a la retícula del ancla (= casillas físicas).
    phys_lp = np.zeros_like(logp)
    phys_vis = np.zeros((len(sel), 64), bool)
    for j in range(len(sel)):
        k = rotation_to_ref(labels[j], ref)
        inv = np.rot90(np.arange(64).reshape(8, 8), k).reshape(64)  # celda j del cuadro -> casilla del ancla
        phys_lp[j, inv] = logp[j]
        phys_vis[j, inv] = labels[j] >= 0
        assert (ref[inv][labels[j] >= 0] == labels[j][labels[j] >= 0]).all()
    res = {"model": a.model.name, "frames": [int(idx[sel].min()), int(idx[sel].max())], "n_frames": int(len(sel))}
    for W in a.windows:
        W_eff = len(sel) if W == 0 else W
        ok = tot = 0
        occ_ok = occ_tot = 0
        for s in range(0, len(sel) - W_eff + 1, max(1, W_eff // 2) if W_eff > 1 else 1):
            acc_lp = (phys_lp[s:s + W_eff] * phys_vis[s:s + W_eff, :, None]).sum(0)
            seen = phys_vis[s:s + W_eff].any(0)
            pred = acc_lp.argmax(-1)
            ok += (pred[seen] == ref[seen]).sum()
            tot += seen.sum()
            m = seen & (ref != 0)
            occ_ok += (pred[m] == ref[m]).sum()
            occ_tot += m.sum()
        key = "todo_el_tramo" if W == 0 else f"ventana_{W}"
        res[key] = {"square_acc": ok / tot, "occupied_acc": occ_ok / occ_tot, "squares_evaluated": int(tot)}
        print(f"{key:14s} casillas {ok / tot:.4f}  ocupadas {occ_ok / occ_tot:.4f}  (n={tot})")
    if 1 in a.windows:
        pred = logits.argmax(-1)
        vis = labels >= 0
        conf = np.zeros((13, 13), int)
        np.add.at(conf, (labels[vis], pred[vis]), 1)
        res["per_class_recall_1frame"] = {c: (float(conf[i, i] / conf[i].sum()) if conf[i].sum() else None)
                                          for i, c in enumerate(g.CLASSES)}
        print({c: (round(v, 2) if v is not None else None) for c, v in res["per_class_recall_1frame"].items()})
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
