"""La referencia reproduce los vectores dorados commiteados."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chessscan import geometry as g

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
CASES = sorted(p for p in GOLDEN.iterdir() if (p / "case.json").exists())


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_case(case):
    meta = json.loads((case / "case.json").read_text())
    H = g.homography(np.array(meta["corners_TL_TR_BR_BL"]))
    assert np.allclose(H, meta["H_row_major"], rtol=1e-9, atol=1e-12)
    canon = g.warp(rgb(case / "image.png"), H)
    assert np.array_equal(canon, rgb(case / "canonical.png"))
    crops = g.crop_all(canon)
    raw = np.fromfile(case / "input_u8.bin", np.uint8).reshape(64, g.CROP_H, g.CROP_W, 3)
    assert np.array_equal(crops, raw)
    for i in (0, 7, 27, 36, 56, 63):
        assert np.array_equal(crops[i], rgb(case / "crops" / f"r{i // 8}c{i % 8}.png"))
