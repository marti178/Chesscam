"""La referencia reproduce los vectores dorados commiteados."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from chessscan import geometry as g

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
CASES = [c for c in sorted(p for p in GOLDEN.iterdir() if (p / "case.json").exists()) if not c.name.startswith("detector")]


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


@pytest.mark.parametrize("variant", [("chess_square_classifier.tflite", "output_logits_f32.bin"),
                                     ("chess_square_classifier_large.tflite", "output_logits_large_f32.bin")],
                         ids=["small", "large"])
@pytest.mark.parametrize("case", CASES, ids=lambda p: p.name)
def test_model_output(case, variant):
    model = GOLDEN.parent / "export" / variant[0]
    if not (case / variant[1]).exists() or not model.exists():
        pytest.skip("sin modelo exportado")
    import tensorflow as tf

    it = tf.lite.Interpreter(model_path=str(model), num_threads=4)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    assert list(inp["shape"]) == [64, g.CROP_H, g.CROP_W, 3] and inp["dtype"] == np.float32
    assert list(out["shape"]) == [64, len(g.CLASSES)] and out["dtype"] == np.float32
    x = np.fromfile(case / "input_u8.bin", np.uint8).reshape(64, g.CROP_H, g.CROP_W, 3).astype(np.float32)
    it.set_tensor(inp["index"], x)
    it.invoke()
    got = it.get_tensor(out["index"])
    want = np.fromfile(case / variant[1], "<f4").reshape(64, len(g.CLASSES))
    assert (got.argmax(1) == want.argmax(1)).all()
    assert np.abs(got - want).max() <= 0.1


DET = GOLDEN / "detector_full_1200x800"


def test_detector_golden():
    model = GOLDEN.parent / "export" / "piece_detector.tflite"
    if not (DET / "output_f32.bin").exists() or not model.exists():
        pytest.skip("falta el detector o su caso dorado")
    import tensorflow as tf
    from chessscan.det_eval import assign, decode, prepare
    case = json.loads((DET / "case.json").read_text())
    img = rgb(DET / case["source_image"])
    c = np.array(case["corners_TL_TR_BR_BL"], float)
    lb, meta = prepare(img, c)
    assert (lb == np.fromfile(DET / "input_letterbox_u8.bin", np.uint8).reshape(lb.shape)).all()
    it = tf.lite.Interpreter(model_path=str(model), num_threads=2)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], lb[None].astype(np.float32) / 255)
    it.invoke()
    y = it.get_tensor(it.get_output_details()[0]["index"])
    want = np.fromfile(DET / "output_f32.bin", "<f4").reshape(y.shape)
    assert np.abs(y - want).max() < 1e-3
    labels = assign(decode(y[0], meta, case["conf"], case["iou"]), c, case["t"]).argmax(-1)
    assert [g.CLASSES[k] for k in labels] == case["squares"]
