import argparse
import os

import cv2
import numpy as np


def build_custom_artag_grid(tag_id: int) -> np.ndarray:
    """Return an 8x8 uint8 grid with values {0,255}.

    Spec:
    - 8x8 cells total
    - 2-cell solid black outer border
    - internal 4x4 region is rows/cols 2..5 (inclusive)
    - orientation marker: white cell at bottom-right of internal 4x4 => (5,5)
    - tag id: 4-bit signature in the central 2x2 (rows/cols 3..4)
      clockwise order with weights 8,4,2,1 matching the repo's decoder:
        (3,3)=8, (3,4)=4, (4,4)=2, (4,3)=1
    """
    if not (0 <= int(tag_id) <= 15):
        raise ValueError("tag_id must be in [0, 15]")

    grid = np.zeros((8, 8), dtype=np.uint8)

    # Orientation marker
    grid[5, 5] = 255
    grid[2,3] = 255
    grid[2,4] = 255
    grid[3,2] = 255
    grid[4,2] = 255
    grid[3,5] = 255
    grid[4,5] = 255
    grid[5,4] = 255
    grid[5,3] = 255
    # Central 2x2 ID bits (clockwise)
    b3 = 1 if (tag_id & 0b1000) else 0
    b2 = 1 if (tag_id & 0b0100) else 0
    b1 = 1 if (tag_id & 0b0010) else 0
    b0 = 1 if (tag_id & 0b0001) else 0

    grid[3, 3] = 255 if b3 else 0  # 8
    grid[3, 4] = 255 if b2 else 0  # 4
    grid[4, 4] = 255 if b1 else 0  # 2
    grid[4, 3] = 255 if b0 else 0  # 1

    return grid


def render_grid(grid: np.ndarray, cell_size: int) -> np.ndarray:
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    grid = np.asarray(grid, dtype=np.uint8)
    if grid.shape != (8, 8):
        raise ValueError("grid must be 8x8")
    return np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the custom 8x8 AR tag used by this project.")
    parser.add_argument("--id", type=int, default=15, help="Tag id in [0,15].")
    parser.add_argument("--cell", type=int, default=100, help="Cell size in pixels.")
    parser.add_argument("--out", type=str, default="assets/custom_tag_id15.png", help="Output PNG path.")
    args = parser.parse_args()

    grid = build_custom_artag_grid(args.id)
    img = render_grid(grid, args.cell)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")

    print(out_path)


if __name__ == "__main__":
    main()
