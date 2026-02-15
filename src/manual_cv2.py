import numpy as np
from numba import njit


@njit(cache=True)
def _cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

@njit(cache=True)
def _convex_hull_jit(points):
    n = len(points)
    if n <= 1:
        return points
    lower = np.zeros((n + 1, 2), dtype=points.dtype)
    k = 0
    for i in range(n):
        while k >= 2 and _cross_product(lower[k-2], lower[k-1], points[i]) <= 0:
            k -= 1
        lower[k] = points[i]
        k += 1
    upper = np.zeros((n + 1, 2), dtype=points.dtype)
    t = 0
    for i in range(n - 1, -1, -1):
        while t >= 2 and _cross_product(upper[t-2], upper[t-1], points[i]) <= 0:
            t -= 1
        upper[t] = points[i]
        t += 1
    result_len = (k - 1) + (t - 1)
    result = np.zeros((result_len, 2), dtype=points.dtype)
    
    result[:k-1] = lower[:k-1]
    result[k-1:] = upper[:t-1]
    
    return result

def convex_hull(points):
    """
    Wrapper to handle sorting and types before JIT kernel
    """
    pts = np.asarray(points, dtype=np.float32)
    ind = np.lexsort((pts[:, 1], pts[:, 0]))
    sorted_pts = pts[ind]
    return _convex_hull_jit(sorted_pts)

def convex_hull_contour(contour):
    pts = _normalize_contour(contour)
    hull = convex_hull(pts)
    return hull.reshape(-1, 1, 2).astype(np.float32)


def _normalize_contour(contour):
    pts = np.asarray(contour)
    if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
        pts = pts[:, 0, :]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("contour must be Nx2 or Nx1x2")
    return pts.astype(np.float64)

def approx_quad(hull, angle_thresh=20):
    hull = np.asarray(hull, dtype=np.float32)
    n = len(hull)
    if n < 4: return None

    corners = []
    for i in range(n):
        prev = hull[(i - 1) % n]
        curr = hull[i]
        nxt  = hull[(i + 1) % n]
        
        ba = prev - curr
        bc = nxt - curr
        norm_ba = np.sqrt(np.sum(ba**2))
        norm_bc = np.sqrt(np.sum(bc**2))
        
        cosang = np.dot(ba, bc) / (norm_ba * norm_bc + 1e-6)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        
        if ang < (180 - angle_thresh):
            corners.append(curr)

    if len(corners) != 4: return None
    return np.array(corners, dtype=np.float32)

def order_corners(corners):
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    idx = np.lexsort((corners[:, 0], corners[:, 1]))[0]
    corners = np.roll(corners, -idx, axis=0)
    return corners.astype(np.float32)

def arc_length(contour, closed=True):
    pts = _normalize_contour(contour)
    if len(pts) < 2: return 0.0
    if closed:
        pts = np.vstack((pts, pts[0]))
    return float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))))


def _rdp(points, epsilon):
    if len(points) < 3: return points
    start, end = points[0], points[-1]
    line = end - start
    line_len_sq = line[0]**2 + line[1]**2
    
    if line_len_sq < 1e-12:
        dists = np.sqrt(np.sum((points - start)**2, axis=1))
    else:
        t = np.sum((points - start) * line, axis=1) / line_len_sq
        proj = start + t[:, None] * line
        dists = np.sqrt(np.sum((points - proj)**2, axis=1))
        
    idx = int(np.argmax(dists))
    max_dist = dists[idx]
    
    if max_dist <= epsilon:
        return np.array([start, end])
    
    left = _rdp(points[:idx+1], epsilon)
    right = _rdp(points[idx:], epsilon)
    return np.vstack((left[:-1], right))

def approx_poly_dp(contour, epsilon, closed=True):
    pts = _normalize_contour(contour)
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack((pts, pts[0]))
        
    approx = _rdp(pts, float(epsilon))
    
    if closed and len(approx) > 1 and np.allclose(approx[0], approx[-1]):
        approx = approx[:-1]
    return approx.reshape(-1, 1, 2).astype(np.float32)

def is_contour_convex(contour):
    pts = _normalize_contour(contour)
    n = len(pts)
    if n < 4: return True
    sign = 0
    for i in range(n):
        p0, p1, p2 = pts[i], pts[(i+1)%n], pts[(i+2)%n]
        cross = (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])
        if cross != 0:
            curr = 1 if cross > 0 else -1
            if sign == 0: sign = curr
            elif sign != curr: return False
    return True


# ==========================================
# 2. IMAGE OPS (JIT OPTIMIZED)
# ==========================================

# @njit(cache=True)
# def _warp_perspective_jit(src_image, H_inv, out_h, out_w):
#     """
#     Loop-based perspective warp.
#     Avoids O(W*H) memory allocation of meshgrids. Faster & Cache friendly.
#     """
#     channels = src_image.shape[2]
#     src_h, src_w = src_image.shape[:2]
    
#     # Output array
#     warped = np.zeros((out_h, out_w, channels), dtype=np.uint8)
    
#     for y in range(out_h):
#         for x in range(out_w):
#             # Apply Homography Inverse
#             # [x, y, 1] @ H_inv.T
#             # H_inv is 3x3
            
#             # Manual matmul for speed
#             src_x_w = H_inv[0,0]*x + H_inv[0,1]*y + H_inv[0,2]
#             src_y_w = H_inv[1,0]*x + H_inv[1,1]*y + H_inv[1,2]
#             w       = H_inv[2,0]*x + H_inv[2,1]*y + H_inv[2,2]
            
#             if abs(w) > 1e-12:
#                 src_x = src_x_w / w
#                 src_y = src_y_w / w
                
#                 # Nearest Neighbor Rounding (floor(x + 0.5))
#                 ix = int(src_x + 0.5)
#                 iy = int(src_y + 0.5)
                
#                 if 0 <= ix < src_w and 0 <= iy < src_h:
#                     warped[y, x] = src_image[iy, ix]
                    
#     return warped
@njit(cache=True)
def _warp_perspective_jit(src_image, H_inv, out_h, out_w):
    
    channels = src_image.shape[2]
    src_h, src_w = src_image.shape[:2]
    warped = np.zeros((out_h, out_w, channels), dtype=np.uint8)
    
    for y in range(out_h):
        for x in range(out_w):
            src_x_w = H_inv[0,0]*x + H_inv[0,1]*y + H_inv[0,2]
            src_y_w = H_inv[1,0]*x + H_inv[1,1]*y + H_inv[1,2]
            w       = H_inv[2,0]*x + H_inv[2,1]*y + H_inv[2,2]
            
            if abs(w) > 1e-9:
                src_x = src_x_w / w
                src_y = src_y_w / w
                x0 = int(src_x)
                y0 = int(src_y)
                x1 = x0 + 1
                y1 = y0 + 1
                if x0 >= 0 and x1 < src_w and y0 >= 0 and y1 < src_h:
                    dx = src_x - x0
                    dy = src_y - y0
                    w00 = (1 - dx) * (1 - dy)
                    w01 = dx * (1 - dy)
                    w10 = (1 - dx) * dy
                    w11 = dx * dy
                    
                    for c in range(channels):
                        val = (w00 * src_image[y0, x0, c] +
                               w01 * src_image[y0, x1, c] +
                               w10 * src_image[y1, x0, c] +
                               w11 * src_image[y1, x1, c])
                        warped[y, x, c] = int(val)
                        
                elif 0 <= int(src_x + 0.5) < src_w and 0 <= int(src_y + 0.5) < src_h:
                     ix = int(src_x + 0.5)
                     iy = int(src_y + 0.5)
                     warped[y, x] = src_image[iy, ix]

    return warped


def warp_perspective(src_image, homography, output_shape):
    H_inv = np.linalg.inv(homography.astype(np.float64))
    return _warp_perspective_jit(src_image, H_inv, int(output_shape[1]), int(output_shape[0]))

def rgb_to_grayscale(image):
    return cv2_like_grayscale(image)

@njit(cache=True)
def cv2_like_grayscale(image):
    h, w = image.shape[:2]
    gray = np.empty((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            val = 0.114 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.299 * image[i, j, 2]
            gray[i, j] = int(val)
    return gray

def threshold_binary(image, threshold_value, max_value=255):
    result = np.zeros_like(image, dtype=np.uint8)
    result[image > threshold_value] = max_value
    return result

def bitwise_not(image): return 255 - image

def bitwise_and_with_mask(image, mask):
    mask_binary = (mask > 0).astype(np.uint8)
    if image.ndim == 3: mask_binary = mask_binary[:, :, None]
    return image * mask_binary

def add_images(image1, image2):
    res = image1.astype(np.int32) + image2.astype(np.int32)
    return np.clip(res, 0, 255).astype(np.uint8)



# @njit(cache=True)
# def _fill_poly_jit(image, poly, color_arr):
    
#     h, w = image.shape[:2]
#     n_pts = len(poly)
    
#     min_y = h
#     max_y = 0
#     for i in range(n_pts):
#         y = poly[i, 1]
#         if y < min_y: min_y = int(y)
#         if y > max_y: max_y = int(y)
            
#     min_y = max(0, min_y)
#     max_y = min(h - 1, max_y)
    
#     # Pre-allocate intersection buffer (max possible intersections is n_pts)
#     # We use a simple list logic
    
#     for y in range(min_y, max_y + 1):
#         # Find intersections for this scanline
#         nodes = []
#         for i in range(n_pts):
#             j = (i + 1) % n_pts
#             y0, x0 = poly[i, 1], poly[i, 0]
#             y1, x1 = poly[j, 1], poly[j, 0]
            
#             if (y0 < y and y1 >= y) or (y1 < y and y0 >= y):
#                 # Calculate x intersection
#                 x = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
#                 nodes.append(x)
        
#         # Sort nodes
#         nodes.sort()
        
#         # Fill pixels between pairs of nodes
#         for i in range(0, len(nodes), 2):
#             if i + 1 >= len(nodes): break
            
#             x_start = int(nodes[i] + 0.9999) # Ceil
#             x_end = int(nodes[i+1])          # Floor
            
#             x_start = max(0, x_start)
#             x_end = min(w - 1, x_end)
            
#             if x_start <= x_end:
#                 image[y, x_start:x_end+1] = color_arr

#     return image

# def fill_convex_poly(image, pts, color):
#     # Wrapper to normalize inputs
#     poly = _normalize_contour(pts)
#     if len(poly) < 3: return image
    
#     # Handle color type
#     if isinstance(color, (int, float)):
#         c = np.array([color], dtype=image.dtype) if image.ndim == 2 else np.full(3, color, dtype=image.dtype)
#     else:
#         c = np.array(color, dtype=image.dtype)
        
#     return _fill_poly_jit(image, poly, c)


