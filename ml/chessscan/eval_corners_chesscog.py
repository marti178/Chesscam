"""Evalúa el detector de esquinas de chesscog (vendor) en data/corners/test, con la métrica de corner_net.

  python -m chessscan.eval_corners_chesscog --per-src 40 --out runs/corners_chesscog.json
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .vendor import chesscog_corners as cc


def corner_error(pred, gt):
    side = np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean() / 8
    return min(max(np.linalg.norm(pred[list(p)] - gt, axis=1)) for p in itertools.permutations(range(4))) / side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/corners"))
    ap.add_argument("--per-src", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cv2.setNumThreads(2)
    items = [json.loads(l) for l in (a.data / "test.jsonl").read_text().splitlines()]
    by = {}
    for it in items:
        by.setdefault(it["src"], []).append(it)
    res = {}
    for src, its in sorted(by.items()):
        errs, ms = [], []
        for it in its[:: max(1, len(its) // a.per_src)][: a.per_src]:
            img = cv2.cvtColor(cv2.imread(str(a.data / it["img"])), cv2.COLOR_BGR2RGB)
            t = time.perf_counter()
            try:
                e = corner_error(np.array(cc.find_corners(cc.CONFIG, img)), np.array(it["corners"]))
            except Exception:
                e = float("inf")
            ms.append((time.perf_counter() - t) * 1000)
            errs.append(e)
        e = np.array(errs)
        res[src] = {"n": len(e), "exito_0.1": float((e < 0.1).mean()), "exito_0.25": float((e < 0.25).mean()),
                    "exito_0.5": float((e < 0.5).mean()), "ms_mediana": float(np.median(ms))}
        print(src, res[src], flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
