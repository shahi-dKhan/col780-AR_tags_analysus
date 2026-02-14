import argparse
import os

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ArUco marker image.")
    parser.add_argument("--id", type=int, default=15, help="Marker id (dictionary dependent).")
    parser.add_argument(
        "--dict",
        dest="dict_name",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name (e.g., DICT_4X4_50).",
    )
    parser.add_argument("--size", type=int, default=800, help="Output image size in pixels.")
    parser.add_argument(
        "--out",
        type=str,
        default="assets/aruco_4x4_50_id15.png",
        help="Output path for the PNG.",
    )
    args = parser.parse_args()

    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco")

    try:
        dict_id = getattr(cv2.aruco, args.dict_name)
    except Exception as exc:
        raise RuntimeError(f"Unknown dictionary: {args.dict_name}") from exc

    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_id = int(args.id)
    size = int(args.size)

    if marker_id < 0:
        raise ValueError("Marker id must be >= 0")
    if size <= 0:
        raise ValueError("Size must be > 0")

    # drawMarker exists in some OpenCV builds; generateImageMarker in newer ones.
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    else:
        marker = cv2.aruco.drawMarker(dictionary, marker_id, size)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ok = cv2.imwrite(out_path, marker)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")

    print(out_path)


if __name__ == "__main__":
    main()
