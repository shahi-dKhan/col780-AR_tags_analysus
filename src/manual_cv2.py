import numpy as np

# ------------------------------------------------------------
# Convex Hull (Monotonic Chain)
# ------------------------------------------------------------

def convex_hull(points):
    """
    points: (N,2) array of (x,y)
    returns hull points in CCW order
    """
    points = sorted(points.tolist())
    if len(points) <= 1:
        return np.array(points, dtype=np.float32)

    def cross(o, a, b):
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float32)


# ------------------------------------------------------------
# Angle-based Quad Approximation (REPLACES approxPolyDP)
# ------------------------------------------------------------

def approx_quad(hull, angle_thresh=20):
    """
    Reduce convex hull to 4 corners by removing near-collinear points
    """
    hull = np.asarray(hull, dtype=np.float32)
    n = len(hull)
    if n < 4:
        return None

    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cosang = np.dot(ba, bc) / (
            np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6
        )
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    corners = []

    for i in range(n):
        prev = hull[(i - 1) % n]
        curr = hull[i]
        nxt  = hull[(i + 1) % n]

        ang = angle(prev, curr, nxt)
        if ang < (180 - angle_thresh):
            corners.append(curr)

    if len(corners) != 4:
        return None

    return np.array(corners, dtype=np.float32)


# ------------------------------------------------------------
# Robust Corner Ordering (TL → TR → BR → BL)
# ------------------------------------------------------------

def order_corners(corners):
    """
    Ensures TL, TR, BR, BL ordering
    """
    center = np.mean(corners, axis=0)

    angles = np.arctan2(
        corners[:, 1] - center[1],
        corners[:, 0] - center[0]
    )
    corners = corners[np.argsort(angles)]

    # Anchor top-left (smallest y, then x)
    idx = np.lexsort((corners[:, 0], corners[:, 1]))[0]
    corners = np.roll(corners, -idx, axis=0)

    return corners.astype(np.float32)


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

# def get_corners(marker_pixels):
#     """
#     marker_pixels: list of (y,x) pixels belonging to tag border
#     returns: (4,2) array of (x,y) corners in TL,TR,BR,BL order
#     """

#     if not marker_pixels or len(marker_pixels) < 20:
#         return None

#     # Convert (y,x) → (x,y)
#     pts = np.array([[p[1], p[0]] for p in marker_pixels], dtype=np.float32)

#     # 1. Convex Hull
#     hull = convex_hull(pts)
#     if len(hull) < 4:
#         return None

#     # 2. Approximate to quadrilateral
#     quad = approx_quad(hull)
#     if quad is None:
#         return None

#     # 3. Order corners
#     ordered = order_corners(quad)

#     return ordered



def warp_perspective(src_image, homography, output_shape):

    out_width, out_height = output_shape
    src_height, src_width = src_image.shape[:2]
    channels = src_image.shape[2]

    H_inv = np.linalg.inv(homography.astype(np.float64))

    y_dst, x_dst = np.meshgrid(
        np.arange(out_height),
        np.arange(out_width),
        indexing="ij"
    )

    ones = np.ones_like(x_dst, dtype=np.float64)
    dst_coords = np.stack([x_dst, y_dst, ones], axis=0).reshape(3, -1).astype(np.float64)

    src_coords = H_inv @ dst_coords

    w = src_coords[2]
    valid_w = np.abs(w) > 1e-12

    x_src = np.zeros_like(w)
    y_src = np.zeros_like(w)

    x_src[valid_w] = src_coords[0, valid_w] / w[valid_w]
    y_src[valid_w] = src_coords[1, valid_w] / w[valid_w]

    # OpenCV-style nearest rounding
    x_nn = np.floor(x_src + 0.5).astype(np.int32)
    y_nn = np.floor(y_src + 0.5).astype(np.int32)

    valid_mask = (
        valid_w &
        (x_nn >= 0) & (x_nn < src_width) &
        (y_nn >= 0) & (y_nn < src_height)
    )

    warped_flat = np.zeros((x_src.shape[0], channels), dtype=np.uint8)

    idx = np.where(valid_mask)[0]

    if idx.size > 0:
        warped_flat[idx] = src_image[y_nn[idx], x_nn[idx]]

    return warped_flat.reshape(out_height, out_width, channels)

def rgb_to_grayscale(image):
    """
    Convert BGR image to grayscale using luminosity method.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be BGR image (H, W, 3)")

    # OpenCV default BGR weights
    weights = np.array([0.114, 0.587, 0.299])
    gray = image @ weights

    return np.clip(gray, 0, 255).astype(np.uint8)


def threshold_binary(image, threshold_value, max_value=255):
    """
    Apply binary threshold.
    """

    if image.ndim != 2:
        raise ValueError("Input must be grayscale image")

    result = np.zeros_like(image, dtype=np.uint8)
    result[image > threshold_value] = max_value

    return result


def bitwise_not(image):
    """
    Bitwise NOT operation.
    """

    return 255 - image


def bitwise_and_with_mask(image, mask):
    """
    Apply binary mask to image.
    """

    if mask.ndim != 2:
        raise ValueError("Mask must be single channel")

    mask_binary = (mask > 0).astype(np.uint8)

    if image.ndim == 3:
        mask_binary = mask_binary[:, :, None]

    return image * mask_binary


def add_images(image1, image2):
    """
    Saturated image addition.
    """

    result = image1.astype(np.int32) + image2.astype(np.int32)
    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


# ------------------------------------------------------------
# Contour utilities (manual replacements for cv2.*)
# ------------------------------------------------------------

def _normalize_contour(contour):
    pts = np.asarray(contour)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts[:, 0, :]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("contour must be Nx2 or Nx1x2")
    return pts.astype(np.float64)


def arc_length(contour, closed=True):
    pts = _normalize_contour(contour)
    if len(pts) < 2:
        return 0.0
    if closed:
        pts = np.vstack([pts, pts[0]])
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def is_contour_convex(contour):
    pts = _normalize_contour(contour)
    n = len(pts)
    if n < 4:
        return True
    sign = 0
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cross != 0:
            curr = 1 if cross > 0 else -1
            if sign == 0:
                sign = curr
            elif sign != curr:
                return False
    return True


def convex_hull_contour(contour):
    pts = _normalize_contour(contour)
    hull = convex_hull(pts)
    return hull.reshape(-1, 1, 2).astype(np.float32)


def _rdp(points, epsilon):
    if len(points) < 3:
        return points
    start = points[0]
    end = points[-1]
    line = end - start
    line_len = np.hypot(line[0], line[1])
    if line_len < 1e-12:
        dists = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    else:
        t = ((points - start) @ line) / (line_len ** 2)
        proj = start + t[:, None] * line
        dists = np.hypot(points[:, 0] - proj[:, 0], points[:, 1] - proj[:, 1])
    idx = int(np.argmax(dists))
    max_dist = dists[idx]
    if max_dist <= epsilon:
        return np.array([start, end])
    left = _rdp(points[: idx + 1], epsilon)
    right = _rdp(points[idx:], epsilon)
    return np.vstack([left[:-1], right])


def approx_poly_dp(contour, epsilon, closed=True):
    pts = _normalize_contour(contour)
    if closed:
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
    approx = _rdp(pts, float(epsilon))
    if closed and len(approx) > 1 and np.allclose(approx[0], approx[-1]):
        approx = approx[:-1]
    return approx.reshape(-1, 1, 2).astype(np.float32)


# ------------------------------------------------------------
# Drawing (manual replacements for cv2.*)
# ------------------------------------------------------------

def fill_convex_poly(image, pts, color):
    poly = _normalize_contour(pts)
    if len(poly) < 3:
        return image
    h, w = image.shape[:2]
    ys = poly[:, 1]
    y_min = max(int(np.floor(np.min(ys))), 0)
    y_max = min(int(np.ceil(np.max(ys))), h - 1)
    edges = list(zip(poly, np.roll(poly, -1, axis=0)))
    for y in range(y_min, y_max + 1):
        x_intersections = []
        for (x0, y0), (x1, y1) in edges:
            if y0 == y1:
                continue
            if (y >= min(y0, y1)) and (y < max(y0, y1)):
                x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                x_intersections.append(x)
        if len(x_intersections) < 2:
            continue
        x_intersections.sort()
        x_start = int(np.ceil(x_intersections[0]))
        x_end = int(np.floor(x_intersections[-1]))
        if x_end < x_start:
            continue
        x_start = max(x_start, 0)
        x_end = min(x_end, w - 1)
        if image.ndim == 2:
            image[y, x_start : x_end + 1] = color
        else:
            image[y, x_start : x_end + 1] = np.array(color, dtype=image.dtype)
    return image


def draw_circle(image, center, radius, color, thickness=-1):
    cx, cy = int(center[0]), int(center[1])
    h, w = image.shape[:2]
    r = max(0, int(radius))
    y0 = max(0, cy - r)
    y1 = min(h - 1, cy + r)
    x0 = max(0, cx - r)
    x1 = min(w - 1, cx + r)
    r2 = r * r
    if thickness < 0:
        for y in range(y0, y1 + 1):
            dy = y - cy
            for x in range(x0, x1 + 1):
                if (x - cx) * (x - cx) + dy * dy <= r2:
                    image[y, x] = color
    else:
        inner = max(0, r - thickness)
        inner2 = inner * inner
        for y in range(y0, y1 + 1):
            dy = y - cy
            for x in range(x0, x1 + 1):
                d2 = (x - cx) * (x - cx) + dy * dy
                if inner2 <= d2 <= r2:
                    image[y, x] = color
    return image


def draw_line(image, pt1, pt2, color, thickness=1):
    x0, y0 = int(pt1[0]), int(pt1[1])
    x1, y1 = int(pt2[0]), int(pt2[1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if thickness <= 1:
            if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
                image[y0, x0] = color
        else:
            draw_circle(image, (x0, y0), thickness // 2, color, thickness=-1)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return image


FONT_HERSHEY_SIMPLEX = 0
LINE_AA = 16

_FONT_5X7 = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def put_text(image, text, org, font_face, font_scale, color, thickness=1, line_type=None):
    scale = max(1, int(round(float(font_scale))))
    x0, y0 = int(org[0]), int(org[1])
    cursor_x = x0
    cursor_y = y0
    for ch in str(text):
        glyph = _FONT_5X7.get(ch, _FONT_5X7[" "])
        for row, pattern in enumerate(glyph):
            for col, bit in enumerate(pattern):
                if bit == "1":
                    px = cursor_x + col * scale
                    py = cursor_y + row * scale
                    for dy in range(scale):
                        for dx in range(scale):
                            if 0 <= py + dy < image.shape[0] and 0 <= px + dx < image.shape[1]:
                                image[py + dy, px + dx] = color
        cursor_x += (5 + 1) * scale
    return image


# ------------------------------------------------------------
# Perspective transform (manual replacement for cv2.perspectiveTransform)
# ------------------------------------------------------------

def perspective_transform_points(points, matrix):
    pts = np.asarray(points)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts[:, 0, :]
    if pts.ndim != 2:
        raise ValueError("points must be Nx3 or Nx2")
    mat = np.asarray(matrix, dtype=np.float64)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros((pts.shape[0],), dtype=np.float64)])
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])
    if mat.shape == (3, 4):
        proj = pts_h @ mat.T
        w = proj[:, 2:3]
        w = np.where(np.abs(w) < 1e-9, 1.0, w)
        out = proj / w
        return out.reshape(-1, 1, 3)
    if mat.shape == (4, 4):
        proj = pts_h @ mat.T
        w = proj[:, 3:4]
        w = np.where(np.abs(w) < 1e-9, 1.0, w)
        out = proj[:, :3] / w
        return out.reshape(-1, 1, 3)
    if mat.shape == (3, 3):
        pts2 = pts[:, :2]
        ones2 = np.ones((pts2.shape[0], 1), dtype=np.float64)
        pts2_h = np.hstack([pts2, ones2])
        proj = pts2_h @ mat.T
        w = proj[:, 2:3]
        w = np.where(np.abs(w) < 1e-9, 1.0, w)
        out = proj / w
        out3 = np.column_stack([out[:, 0], out[:, 1], np.zeros_like(out[:, 0])])
        return out3.reshape(-1, 1, 3)
    raise ValueError("Unsupported perspective matrix shape")

