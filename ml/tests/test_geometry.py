"""Pruebas de la referencia contra las tablas de CONTRACT.md."""
import numpy as np

from chessscan import geometry as g


def test_tables_match_contract():
    assert [g.row_bounds(r) for r in range(8)] == [
        (0, 128), (42, 160), (83, 192), (124, 224), (165, 256), (206, 288), (247, 320), (288, 352)]
    assert [g.col_bounds(c) for c in range(8)] == [
        (0, 64), (40, 96), (80, 128), (120, 160), (160, 200), (192, 240), (224, 280), (256, 320)]


def test_square_lands_bottom_left():
    rng = np.random.default_rng(0)
    canon = rng.integers(0, 256, (g.CH, g.CW, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            crop = g.crop_square(canon, r, c)
            sq = canon[g.MY + g.S * r: g.MY + g.S * (r + 1), g.MX + g.S * c: g.MX + g.S * (c + 1)]
            if c < 4:
                sq = sq[:, ::-1]
            assert np.array_equal(crop[96:128, 0:32], sq)


def test_homography_maps_corners():
    corners = np.array([[350, 237], [805, 216], [1001, 627], [241, 686]], float)
    H = g.homography(corners)
    p = np.c_[corners, np.ones(4)] @ H.T
    assert np.allclose(p[:, :2] / p[:, 2:], g.DST_CORNERS, atol=1e-6)


def test_labels_orientation():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    lab = [g.CLASSES[i] for i in g.board_labels(start, "BL")]
    assert lab[56:64] == list("RNBQKBNR") and lab[0:8] == list("rnbqkbnr")
    lab = [g.CLASSES[i] for i in g.board_labels(start, "TR")]
    assert lab[0:8] == list("RNBKQBNR")
    lab = [g.CLASSES[i] for i in g.board_labels(start, "TL")]
    assert [lab[r * 8] for r in range(8)] == list("RNBQKBNR")


def test_sort_corners():
    pts = np.array([[805, 216], [1001, 627], [241, 686], [350, 237]], float)
    assert g.sort_corners(pts).tolist() == [[350, 237], [805, 216], [1001, 627], [241, 686]]
