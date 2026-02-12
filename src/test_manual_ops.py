import numpy as np
from manual_cv2 import (
    arc_length,
    approx_poly_dp,
    convex_hull_contour,
    draw_circle,
    draw_line,
    fill_convex_poly,
    is_contour_convex,
    perspective_transform_points,
    put_text,
)


def test_contour_ops():
    contour = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [2, 2]], dtype=np.float32).reshape(-1, 1, 2)
    hull = convex_hull_contour(contour)
    assert len(hull) == 4
    perim = arc_length(hull, True)
    assert perim > 0
    approx = approx_poly_dp(hull, 0.01 * perim, True)
    assert len(approx) == 4
    assert is_contour_convex(approx)


def test_fill_and_draw():
    img = np.zeros((20, 20), dtype=np.uint8)
    pts = np.array([[5, 5], [14, 5], [14, 14], [5, 14]], dtype=np.int32)
    fill_convex_poly(img, pts, 255)
    assert img[10, 10] == 255
    assert img[1, 1] == 0

    color = (0, 255, 0)
    img_color = np.zeros((20, 20, 3), dtype=np.uint8)
    draw_line(img_color, (0, 0), (19, 19), color, 1)
    assert img_color.sum() > 0

    draw_circle(img_color, (10, 10), 3, (255, 0, 0), -1)
    assert img_color[10, 10, 0] == 255


def test_text_and_transform():
    img = np.zeros((20, 60, 3), dtype=np.uint8)
    put_text(img, "FPS: 1.0", (2, 2), 0, 1.0, (255, 255, 255), 1, 0)
    assert img.sum() > 0

    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64).reshape(-1, 1, 3)
    mat = np.eye(3, 4, dtype=np.float64)
    out = perspective_transform_points(pts, mat)
    assert out.shape[0] == 3


def run_tests():
    test_contour_ops()
    test_fill_and_draw()
    test_text_and_transform()
    print("manual_cv2 tests passed")


if __name__ == "__main__":
    run_tests()
