"""Recupera las fotos de ChessReD que no traen esquinas (fuera de chessred2k), verificadas contra su FEN.

Para cada foto: esquinas con el detector entrenado (`corner_detector.tflite`, dos pasadas), canónica,
clasificador v6l una vez, y se prueban las 4 posiciones posibles de a1 contra las piezas anotadas.
Se acepta solo si (1) la mejor posición acierta ≥ --min-agree casillas de 64, (2) le saca ≥ --margin
casillas a la segunda y (3) si hay cajas de piezas, su base cae en su casilla (mismo control que
`prepare_chessred`; fuera de chessred2k no hay cajas). Solo los splits train y val completos: el test completo queda afuera, porque
comparte partidas con el test de chessred2k.

  python -m chessscan.recover_chessred --out data/prepared_chessred_auto
Salida por split: canon.npy, labels.npy, names.txt (como prepare_chessred) y log.jsonl con cada decisión.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from . import geometry as g
from .corner_net import predict_zoom
from .prepare_chessred import CAT, find_image


def tflite(path: Path, threads: int):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    it.allocate_tensors()
    return it


def detector(path: Path, threads: int):
    it = tflite(path, threads)
    inp = it.get_input_details()[0]["index"]
    names = {("heat" if d["shape"][-1] == 1 else "off"): d["index"] for d in it.get_output_details()}

    def pred(x):
        it.set_tensor(inp, x)
        it.invoke()
        return {k: it.get_tensor(i) for k, i in names.items()}
    return pred


def classifier(path: Path, threads: int):
    it = tflite(path, threads)
    inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

    def pred(canon):
        it.set_tensor(inp, g.model_input(g.crop_all(canon)))
        it.invoke()
        return it.get_tensor(out).argmax(1)
    return pred


def labels_for(pieces, a1: str) -> np.ndarray:
    sq_of = {g.grid_to_square(r, c, a1): r * 8 + c for r in range(8) for c in range(8)}
    lab = np.zeros(64, np.int64)
    for p in pieces:
        lab[sq_of[p["chessboard_position"]]] = g.CLASS_TO_INDEX[CAT[p["category_id"]]]
    return lab


def boxes_bad(pieces, a1: str, H: np.ndarray, k: np.ndarray) -> int:
    sq_of = {g.grid_to_square(r, c, a1): r * 8 + c for r in range(8) for c in range(8)}
    bad = 0
    for p in pieces:
        if "bbox" not in p:  # fuera de chessred2k no hay cajas
            continue
        i = sq_of[p["chessboard_position"]]
        x, y, w, h = p["bbox"]
        q = cv2.perspectiveTransform(np.float32([[[(x + w / 2) * k[0], (y + h) * k[1]]]]), H)[0, 0]
        r_est, c_est = (q[1] - g.MY) / g.S, (q[0] - g.MX) / g.S
        if abs(c_est - (i % 8 + 0.5)) > 1.2 or not (-0.5 < r_est - i // 8 < 2.2):
            bad += 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=Path("C:/Users/MartinM/ml_fotos/chessred/annotations.json"))
    ap.add_argument("--images", type=Path, default=Path("data/raw/chessred"))
    ap.add_argument("--out", type=Path, default=Path("data/prepared_chessred_auto"))
    ap.add_argument("--detector", type=Path, default=Path("export/corner_detector.tflite"))
    ap.add_argument("--classifier", type=Path, default=Path("export/chess_square_classifier_large.tflite"))
    ap.add_argument("--min-agree", type=int, default=60)
    ap.add_argument("--margin", type=int, default=6)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=0, help="solo las primeras N por split (prueba)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    ann = json.loads(a.ann.read_text())
    images = {im["id"]: im for im in ann["images"]}
    has_corners = {c["image_id"] for c in ann["annotations"]["corners"]}
    pieces: dict[int, list] = {}
    for p in ann["annotations"]["pieces"]:
        if p["category_id"] != 12:
            pieces.setdefault(p["image_id"], []).append(p)
    det, cls = detector(a.detector, a.threads), classifier(a.classifier, a.threads)
    for split in a.splits:
        ids = [i for i in ann["splits"][split]["image_ids"] if i not in has_corners]
        ids = ids[:a.limit] if a.limit else ids
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        canon = np.lib.format.open_memmap(d / "canon_tmp.npy", "w+", np.uint8, (len(ids), g.CH, g.CW, 3))
        labels, names, n = [], [], 0
        stats: dict[str, int] = {}
        with open(d / "log.jsonl", "w") as log:
            for iid in tqdm(ids, desc=split):
                im = images[iid]
                path = find_image(a.images, im["path"], im["file_name"])
                rec = {"id": iid, "file": im["file_name"]}
                if path is None:
                    rec["motivo"] = "no está la imagen"
                else:
                    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    k = np.array([img.shape[1] / im["width"], img.shape[0] / im["height"]])
                    pts, sc = predict_zoom(det, img)
                    if len(pts) < 4:
                        rec["motivo"] = "menos de 4 esquinas"
                    else:
                        c4 = g.sort_corners(pts)
                        H = g.homography(c4)
                        cv = g.warp(img, H)
                        pred = cls(cv)
                        ps = pieces.get(iid, [])
                        agree = {a1: int((pred == labels_for(ps, a1)).sum()) for a1 in ("TL", "TR", "BR", "BL")}
                        best, second = sorted(agree.values(), reverse=True)[:2]
                        a1 = max(agree, key=agree.get)
                        bad = boxes_bad(ps, a1, H, k)
                        rec.update(acierto=agree, a1=a1, cajas_mal=bad, p_min=float(min(sc[:4])), corners=np.round(c4, 1).tolist())
                        if best < a.min_agree:
                            rec["motivo"] = "no coincide con el FEN"
                        elif best - second < a.margin:
                            rec["motivo"] = "orientación ambigua"
                        elif ps and bad > max(2, 0.2 * len(ps)):
                            rec["motivo"] = "cajas fuera de su casilla"
                        else:
                            rec["motivo"] = "ok"
                            canon[n] = cv
                            labels.append(labels_for(ps, a1))
                            names.append(f"{im['file_name']} a1={a1}")
                            n += 1
                stats[rec["motivo"]] = stats.get(rec["motivo"], 0) + 1
                log.write(json.dumps(rec) + "\n")
        out = np.lib.format.open_memmap(d / "canon.npy", "w+", np.uint8, (n, g.CH, g.CW, 3))
        for i in range(0, n, 256):
            out[i:i + 256] = canon[i:min(i + 256, n)]
        out.flush()
        del out, canon
        (d / "canon_tmp.npy").unlink()
        np.save(d / "labels.npy", np.stack(labels) if labels else np.zeros((0, 64), np.int64))
        (d / "names.txt").write_text("\n".join(names) + "\n")
        (d / "stats.json").write_text(json.dumps({"candidatas": len(ids), **stats}, indent=1))
        print(split, {"candidatas": len(ids), **stats}, flush=True)


if __name__ == "__main__":
    main()
