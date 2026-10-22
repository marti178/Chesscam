"""Verifica el FEN de un video autoetiquetado contra lo que predice un modelo.

Compara, en los cuadros autoetiquetados, la predicción del modelo con las
etiquetas del FEN dado y con las de FEN alternativos (por ejemplo, el mismo FEN
girado 180°, que es lo que pasa si el usuario lo leyó desde el lado de las
negras). Las etiquetas alternativas se arman rotando la retícula de cada cuadro
igual que la del FEN dado, así que miden la misma geometría.

  python -m chessscan.verify_fen --model export/chess_square_classifier.tflite --data data/real/video3 \
      --alt "2k2b1r/1bp2p2/7p/6p1/Q7/2P1B2P/PPB2PPK/RN2q3" --every 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry as g
from .data import Boards
from .eval_video import rotation_to_ref
from .pipeline import tflite_predict


def rot180_fen(fen: str) -> str:
    rows = fen.split()[0].split("/")
    return "/".join(r[::-1] for r in rows[::-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--alt", nargs="*", default=[], help="FEN alternativos (además del girado 180°)")
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    meta = json.loads((a.data / "frames.json").read_text())
    fen, a1 = meta["fen"], meta["a1_anchor"]
    hyps = {"dado": fen, "girado_180": rot180_fen(fen)}
    for i, f in enumerate(a.alt):
        hyps[f"alt{i}"] = f
    ref = g.board_labels(fen, a1)
    b = Boards.load(a.data)
    sel = np.arange(0, len(b), a.every)
    x = np.concatenate([g.crop_all(np.asarray(b.canon[i])) for i in sel])
    pred = tflite_predict(a.model, x).argmax(1).reshape(len(sel), 64)
    res = {}
    for name, f in hyps.items():
        R = g.board_labels(f, a1).reshape(8, 8)
        ok = tot = occ_ok = occ_tot = 0
        for j, i in enumerate(sel):
            lab = b.labels[i]
            k = rotation_to_ref(lab, ref)
            L = np.rot90(R, k).reshape(64)
            vis = lab >= 0
            ok += (pred[j][vis] == L[vis]).sum()
            tot += vis.sum()
            m = vis & (L != 0)
            occ_ok += (pred[j][m] == L[m]).sum()
            occ_tot += m.sum()
        res[name] = {"fen": f, "square_acc": ok / tot, "occupied_acc": occ_ok / max(1, occ_tot)}
        print(f"{name:11s} casillas {ok / tot:.3f}  piezas {occ_ok / max(1, occ_tot):.3f}  {f}")
    (a.data / "verify_fen.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
