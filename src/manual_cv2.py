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

def get_corners(marker_pixels):
    """
    marker_pixels: list of (y,x) pixels belonging to tag border
    returns: (4,2) array of (x,y) corners in TL,TR,BR,BL order
    """

    if not marker_pixels or len(marker_pixels) < 20:
        return None

    # Convert (y,x) → (x,y)
    pts = np.array([[p[1], p[0]] for p in marker_pixels], dtype=np.float32)

    # 1. Convex Hull
    hull = convex_hull(pts)
    if len(hull) < 4:
        return None

    # 2. Approximate to quadrilateral
    quad = approx_quad(hull)
    if quad is None:
        return None

    # 3. Order corners
    ordered = order_corners(quad)

    return ordered



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
