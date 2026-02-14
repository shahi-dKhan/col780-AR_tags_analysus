import argparse
import os

import cv2
import numpy as np


def _draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int = 255) -> None:
    h, w = img.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1] = value


def generate_cross_marker(
    width: int = 768,
    height: int = 886,
    central_size_ratio: float = 0.46,
    top_ext_w_ratio: float = 0.24,
    top_ext_h_ratio: float = 0.12,
    side_ext_w_ratio: float = 0.12,
    side_ext_h_ratio: float = 0.22,
    center_y_shift_ratio: float = -0.06,
    bottom_line_margin_ratio: float = 0.03,
    bottom_line_thickness: int = 3,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    img = np.zeros((height, width), dtype=np.uint8)

    cx = width // 2
    cy = int(height * 0.5 + center_y_shift_ratio * height)

    central = int(round(width * central_size_ratio))
    central = max(10, min(central, min(width, height) - 10))

    top_w = int(round(width * top_ext_w_ratio))
    top_h = int(round(height * top_ext_h_ratio))
    side_w = int(round(width * side_ext_w_ratio))
    side_h = int(round(height * side_ext_h_ratio))

    # Central square
    x0 = cx - central // 2
    x1 = x0 + central
    y0 = cy - central // 2
    y1 = y0 + central
    _draw_rect(img, x0, y0, x1, y1, 255)

    # Top extension (centered)
    tx0 = cx - top_w // 2
    tx1 = tx0 + top_w
    ty1 = y0
    ty0 = ty1 - top_h
    _draw_rect(img, tx0, ty0, tx1, ty1, 255)

    # Side extensions (centered vertically on the central square)
    sy0 = cy - side_h // 2
    sy1 = sy0 + side_h

    # Left extension
    lx1 = x0
    lx0 = lx1 - side_w
    _draw_rect(img, lx0, sy0, lx1, sy1, 255)

    # Right extension
    rx0 = x1
    rx1 = rx0 + side_w
    _draw_rect(img, rx0, sy0, rx1, sy1, 255)

    # Bottom horizontal line
    margin = int(round(height * bottom_line_margin_ratio))
    y_line = height - margin - bottom_line_thickness
    _draw_rect(img, 0, y_line, width, y_line + bottom_line_thickness, 255)

    # Return as 3-channel for consistency with cv2.imread usage in overlay
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the cross marker template (black bg, white cross, bottom line).")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=886)
    parser.add_argument("--out", type=str, default="assets/cross_marker.png")

    args = parser.parse_args()
    out_path = args.out

    img = generate_cross_marker(width=args.width, height=args.height)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")

    print(out_path)


if __name__ == "__main__":
    main()
