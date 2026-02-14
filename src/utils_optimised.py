import cv2
import numpy as np
from numba import njit
# Import your manual optimized functions (only the ones actually used)
from manual_optimised import (
    convex_hull_contour,
    approx_poly_dp,
    arc_length,
    # fill_convex_poly,
    convex_hull,
    warp_perspective,
)


def generate_tag(cell_size=50, tag_id=0):
    """
    Generate an AR tag image with the specified ID.
    """
    # Initialize an 8x8 black grid (0 = black)
    # The 2-cell outer border is already black by default
    grid = np.zeros((8, 8), dtype=np.uint8)
    
    # Define the internal 4x4 grid (Indices 2 to 5)
    # Row 2
    grid[2, 2] = 0
    grid[2, 3] = 255
    grid[2, 4] = 255
    grid[2, 5] = 0
    
    # Row 3
    grid[3, 2] = 255
    grid[3, 3] = 255  # ID Bit 1
    grid[3, 4] = 0  # ID Bit 2
    grid[3, 5] = 255
    
    # Row 4
    grid[4, 2] = 255
    grid[4, 3] = 255  # ID Bit 4
    grid[4, 4] = 255  # ID Bit 3
    grid[4, 5] = 255
    
    # Row 5
    grid[5, 2] = 0
    grid[5, 3] = 255
    grid[5, 4] = 255
    grid[5, 5] = 255

    # Scale the 8x8 grid to a visible image size
    tag_image = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    
    cv2.imwrite(f"Tag{tag_id}.png", tag_image)

    return tag_image




@njit(cache=True)
def fast_component_labeling_local(binary_img, roi_offset_y, roi_offset_x):
    h, w = binary_img.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    max_area = 0
    best_pixels_y = [np.int32(0)]; best_pixels_y.pop()
    best_pixels_x = [np.int32(0)]; best_pixels_x.pop()
    
    stack_y = np.zeros(h * w, dtype=np.int32)
    stack_x = np.zeros(h * w, dtype=np.int32)
    
    for y in range(h):
        for x in range(w):
            if binary_img[y, x] == 0 and not visited[y, x]:
                stack_ptr = 0
                stack_y[stack_ptr] = y
                stack_x[stack_ptr] = x
                stack_ptr += 1
                visited[y, x] = True
                current_count = 0
                touches_border = False
                curr_comp_y = []
                curr_comp_x = []
                while stack_ptr > 0:
                    stack_ptr -= 1
                    cy = stack_y[stack_ptr]
                    cx = stack_x[stack_ptr]
                    curr_comp_y.append(cy)
                    curr_comp_x.append(cx)
                    current_count += 1
                    if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                        touches_border = True
                    if cy > 0 and binary_img[cy - 1, cx] == 0 and not visited[cy - 1, cx]:
                        visited[cy - 1, cx] = True; stack_y[stack_ptr] = cy - 1; stack_x[stack_ptr] = cx; stack_ptr += 1
                    if cy < h - 1 and binary_img[cy + 1, cx] == 0 and not visited[cy + 1, cx]:
                        visited[cy + 1, cx] = True; stack_y[stack_ptr] = cy + 1; stack_x[stack_ptr] = cx; stack_ptr += 1
                    if cx > 0 and binary_img[cy, cx - 1] == 0 and not visited[cy, cx - 1]:
                        visited[cy, cx - 1] = True; stack_y[stack_ptr] = cy; stack_x[stack_ptr] = cx - 1; stack_ptr += 1
                    if cx < w - 1 and binary_img[cy, cx + 1] == 0 and not visited[cy, cx + 1]:
                        visited[cy, cx + 1] = True; stack_y[stack_ptr] = cy; stack_x[stack_ptr] = cx + 1; stack_ptr += 1
                
                if not touches_border and current_count > max_area:
                    max_area = current_count
                    best_pixels_y = curr_comp_y
                    best_pixels_x = curr_comp_x
                    
    final_pixels = []
    for i in range(len(best_pixels_y)):
        final_pixels.append((best_pixels_y[i] + roi_offset_y, best_pixels_x[i] + roi_offset_x))
    return final_pixels



def threshold_image(frame):
    gray = 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
    gray = gray.astype(np.uint8)
    thresh_val = 150
    binary = np.zeros_like(gray)
    binary[gray >= thresh_val] = 255
    return binary


def split_ROI(binary_image, min_sheet_area=500, scale=8):
    from collections import deque
    h, w = binary_image.shape
    h_small, w_small = h // scale, w // scale
    downsampled = binary_image[::scale, ::scale]
    visited = np.zeros((h_small, w_small), dtype=np.bool_)
    islands = []
    min_area_small = max(1, min_sheet_area // (scale * scale))
    white_pixel_coords = np.argwhere(downsampled == 255) 
    for start_y, start_x in white_pixel_coords:
        if visited[start_y, start_x]: continue
        island = []
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            island.append((y * scale, x * scale))
            if y > 0 and downsampled[y-1, x] == 255 and not visited[y-1, x]:
                visited[y-1, x] = True; queue.append((y-1, x))
            if y < h_small-1 and downsampled[y+1, x] == 255 and not visited[y+1, x]:
                visited[y+1, x] = True; queue.append((y+1, x))
            if x > 0 and downsampled[y, x-1] == 255 and not visited[y, x-1]:
                visited[y, x-1] = True; queue.append((y, x-1))
            if x < w_small-1 and downsampled[y, x+1] == 255 and not visited[y, x+1]:
                visited[y, x+1] = True; queue.append((y, x+1))
        if len(island) >= min_area_small:
            islands.append(island)
    return islands

def detect_tag(image, island, gray_image, scale=8):
    h, w = gray_image.shape
    island_arr = np.array(island, dtype=np.int32)
    min_y, max_y = island_arr[:, 0].min(), island_arr[:, 0].max()
    min_x, max_x = island_arr[:, 1].min(), island_arr[:, 1].max()
    for _ in range(scale + 2):
        if min_y > 0 and np.any(gray_image[min_y - 1, min_x : max_x + 1] == 255): min_y -= 1
        else: break
    for _ in range(scale + 2):
        if max_y < h - 1 and np.any(gray_image[max_y + 1, min_x : max_x + 1] == 255): max_y += 1
        else: break
    for _ in range(scale + 2):
        if min_x > 0 and np.any(gray_image[min_y : max_y + 1, min_x - 1] == 255): min_x -= 1
        else: break
    for _ in range(scale + 2):
        if max_x < w - 1 and np.any(gray_image[min_y : max_y + 1, max_x + 1] == 255): max_x += 1
        else: break
    pad = 1
    safe_min_y = max(0, min_y - pad)
    safe_max_y = min(h, max_y + 1 + pad)
    safe_min_x = max(0, min_x - pad)
    safe_max_x = min(w, max_x + 1 + pad)
    
    roi = gray_image[safe_min_y : safe_max_y, safe_min_x : safe_max_x]
    tag_pixels = fast_component_labeling_local(roi, safe_min_y, safe_min_x)
    if not tag_pixels:
        return []
    roi_h, roi_w = roi.shape
    mask = np.zeros((roi_h, roi_w), dtype=np.bool_)
    local_y = [p[0] - safe_min_y for p in tag_pixels]
    local_x = [p[1] - safe_min_x for p in tag_pixels]
    mask[local_y, local_x] = True
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & 
              padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary_mask = mask ^ eroded
    by, bx = np.where(boundary_mask)
    marker_pixels = []
    for i in range(len(by)):
        marker_pixels.append((by[i] + safe_min_y, bx[i] + safe_min_x))
    return marker_pixels

def get_corners(marker_pixels):
    if not marker_pixels or len(marker_pixels) < 4: return None
    pts = np.array([[p[1], p[0]] for p in marker_pixels], dtype=np.float32)
    hull = convex_hull(pts) # From manual_cv2
    perimeter = arc_length(hull, True)
    corners = None

    for eps_factor in np.linspace(0.01, 0.08, 8):
        approx = approx_poly_dp(hull, eps_factor * perimeter, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            break

    if corners is None: return None
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    idx = np.lexsort((corners[:, 0], corners[:, 1]))[0]
    corners = np.roll(corners, -idx, axis=0)
    return corners.astype(np.float32)

class ARtag:
    def __init__(self, corners, id, orientation_steps):
        self.corners = corners
        self.id = id
        self.orientation_steps = orientation_steps
        self.rotation = orientation_steps * 90

def compute_homography_manual(src_pts, dst_pts):
    """Solves Homography using SVD (No OpenCV)"""
    A = []
    for i in range(4):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def perspective_transform_manual(points, matrix):
    """Project points using matrix (No OpenCV)"""
    pts = points.reshape(-1, 3)
    projected = matrix @ pts.T # (3, N)
    projected = projected.T # (N, 3)
    z = projected[:, 2:3]
    z[np.abs(z) < 1e-9] = 1.0 # Avoid div zero
    out = projected[:, :2] / z
    return out

def decode_tag(marker_pixels, gray_image): 
    corners = get_corners(marker_pixels)
    if corners is None: return None
    src_pts = np.array([[0,0],[8,0],[8,8],[0,8]], dtype=np.float32)
    dst_pts = corners.astype(np.float32)
    H = compute_homography_manual(src_pts, dst_pts)
    sample_resolution = 5
    margin = 0.2 
    offsets = np.linspace(-0.5 + margin, 0.5 - margin, sample_resolution)
    grid_vals = np.zeros((8, 8), dtype=np.uint8)
    h, w = gray_image.shape

    for i in range(8):       # Columns
        for j in range(8):   # Rows
            cell_samples = []
            for dy in offsets:
                for dx in offsets:
                    u, v = i + 0.5 + dx, j + 0.5 + dy
                    
                    denom = H[2, 0]*u + H[2, 1]*v + H[2, 2]
                    if abs(denom) < 1e-6: continue 
                    
                    x_img = (H[0, 0]*u + H[0, 1]*v + H[0, 2]) / denom
                    y_img = (H[1, 0]*u + H[1, 1]*v + H[1, 2]) / denom
                    
                    if 0 <= int(y_img) < h and 0 <= int(x_img) < w:
                        cell_samples.append(gray_image[int(y_img), int(x_img)])

            # MEDIAN VOTE (robust to noise)
            if len(cell_samples) > 0:
                cell_samples.sort()
                median_pixel = cell_samples[len(cell_samples) // 2]
                grid_vals[j, i] = 1 if median_pixel > 127 else 0

    
    if grid_vals[5, 5] == 1: rotation = 0
    elif grid_vals[5, 2] == 1: rotation = 1
    elif grid_vals[2, 2] == 1: rotation = 2
    elif grid_vals[2, 5] == 1: rotation = 3
    else: return None
    
    if rotation > 0:
        grid_vals = np.rot90(grid_vals, k=rotation)
        corners = np.roll(corners, shift=-rotation, axis=0)

    found_id = (grid_vals[3,3]*8) + (grid_vals[3,4]*4) + (grid_vals[4,4]*2) + (grid_vals[4,3]*1)
    return ARtag(corners, found_id, rotation)

def compute_projection_matrix(corners, K):
    src_pts = np.array([[0,0], [200,0], [200,200], [0,200]], dtype=np.float32)
    H = compute_homography_manual(src_pts, corners)
    try:
        inv_K = np.linalg.inv(K)
        A = inv_K @ H
        col1 = A[:, 0]; col2 = A[:, 1]; col3 = A[:, 2]
        norm = (np.linalg.norm(col1) + np.linalg.norm(col2)) / 2.0
        if norm == 0: return None
        r1 = col1 / norm; r2 = col2 / norm; t = col3 / norm
        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        extrinsics = np.column_stack([R, t])
        P = K @ extrinsics
        return P
    except: return None


class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.faces = []
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz: v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'f':
                face = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                self.faces.append(face)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, scale=3, color=False):
    """
    Render a loaded obj model into the current video frame.

    Args:
        img: The current video frame.
        obj: The loaded OBJ model.
        projection: The 3D projection matrix.
        scale: Scale factor for the 3D model. Defaults to 3.
        color: Whether to render in color. Defaults to False.
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    h, w = 200, 200  # AR tag reference size

    for face in obj.faces:
        face_vertices = face  # face is already a list of vertex indices
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)

    return img

def read_intrinsics(path):
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
    try:
        with open(path, 'r') as f:
            content = f.read()
            import re
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
            if len(nums) >= 9:
                K = np.array(nums[:9], dtype=np.float32).reshape(3, 3)
    except: pass
    return K


_obj_cache = {}

def process_frame_3D(frame, model_path, intrinsics_path, scale_3d=50.0, scale=4):
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    K = read_intrinsics(intrinsics_path)
    if model_path not in _obj_cache:
        _obj_cache[model_path] = OBJ(model_path, swapyz=True)
    obj = _obj_cache[model_path]
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                P = compute_projection_matrix(tag.corners, K)
                if P is not None:
                    frame = render(frame, obj, P, scale=scale_3d)
    return frame

def process_frame_marking(frame, scale=4):
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    detected_tags = []
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                detected_tags.append(tag)
    
    for tag in detected_tags:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        for i, point in enumerate(tag.corners):
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 8, colors[i], -1)
            next_point = tag.corners[(i + 1) % 4]
            nx, ny = int(next_point[0]), int(next_point[1])
            cv2.line(frame, (x,y), (nx,ny), (0,255,0), 2)
        
        cx = int(sum(c[0] for c in tag.corners) / 4)
        cy = int(sum(c[1] for c in tag.corners) / 4)
        text = f"ID:{tag.id} {tag.rotation}deg"
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    return frame

def process_frame_superimpose(frame, template_path, scale=4):
    template = cv2.imread(template_path)
    if template is None: return frame
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    h_temp, w_temp = template.shape[:2]
    src_pts = np.array([[0,0], [w_temp,0], [w_temp,h_temp], [0,h_temp]], dtype=np.float32)
    
    frame_h, frame_w = frame.shape[:2]
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                H = compute_homography_manual(src_pts, tag.corners)
                warped = warp_perspective(template, H, (frame_w, frame_h))
                mask_src = np.ones((h_temp, w_temp, 1), dtype=np.uint8) * 255
                mask_warped = warp_perspective(mask_src, H, (frame_w, frame_h))
                mask_bool = mask_warped[:, :, 0] > 0
                frame[mask_bool] = warped[mask_bool]
                
    return frame

import cv2
import numpy as np
from numba import njit
# Import your manual optimized functions (only the ones actually used)
from manual_optimised import (
    convex_hull_contour,
    approx_poly_dp,
    arc_length,
    # fill_convex_poly,
    convex_hull,
    warp_perspective,
)


def generate_tag(cell_size=50, tag_id=0):
    """
    Generate an AR tag image with the specified ID.
    """
    # Initialize an 8x8 black grid (0 = black)
    # The 2-cell outer border is already black by default
    grid = np.zeros((8, 8), dtype=np.uint8)
    
    # Define the internal 4x4 grid (Indices 2 to 5)
    # Row 2
    grid[2, 2] = 0
    grid[2, 3] = 255
    grid[2, 4] = 255
    grid[2, 5] = 0
    
    # Row 3
    grid[3, 2] = 255
    grid[3, 3] = 255  # ID Bit 1
    grid[3, 4] = 0  # ID Bit 2
    grid[3, 5] = 255
    
    # Row 4
    grid[4, 2] = 255
    grid[4, 3] = 255  # ID Bit 4
    grid[4, 4] = 255  # ID Bit 3
    grid[4, 5] = 255
    
    # Row 5
    grid[5, 2] = 0
    grid[5, 3] = 255
    grid[5, 4] = 255
    grid[5, 5] = 255

    # Scale the 8x8 grid to a visible image size
    tag_image = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    
    cv2.imwrite(f"Tag{tag_id}.png", tag_image)

    return tag_image




@njit(cache=True)
def fast_component_labeling_local(binary_img, roi_offset_y, roi_offset_x):
    h, w = binary_img.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    max_area = 0
    best_pixels_y = [np.int32(0)]; best_pixels_y.pop()
    best_pixels_x = [np.int32(0)]; best_pixels_x.pop()
    
    stack_y = np.zeros(h * w, dtype=np.int32)
    stack_x = np.zeros(h * w, dtype=np.int32)
    
    for y in range(h):
        for x in range(w):
            if binary_img[y, x] == 0 and not visited[y, x]:
                stack_ptr = 0
                stack_y[stack_ptr] = y
                stack_x[stack_ptr] = x
                stack_ptr += 1
                visited[y, x] = True
                current_count = 0
                touches_border = False
                curr_comp_y = []
                curr_comp_x = []
                while stack_ptr > 0:
                    stack_ptr -= 1
                    cy = stack_y[stack_ptr]
                    cx = stack_x[stack_ptr]
                    curr_comp_y.append(cy)
                    curr_comp_x.append(cx)
                    current_count += 1
                    if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                        touches_border = True
                    if cy > 0 and binary_img[cy - 1, cx] == 0 and not visited[cy - 1, cx]:
                        visited[cy - 1, cx] = True; stack_y[stack_ptr] = cy - 1; stack_x[stack_ptr] = cx; stack_ptr += 1
                    if cy < h - 1 and binary_img[cy + 1, cx] == 0 and not visited[cy + 1, cx]:
                        visited[cy + 1, cx] = True; stack_y[stack_ptr] = cy + 1; stack_x[stack_ptr] = cx; stack_ptr += 1
                    if cx > 0 and binary_img[cy, cx - 1] == 0 and not visited[cy, cx - 1]:
                        visited[cy, cx - 1] = True; stack_y[stack_ptr] = cy; stack_x[stack_ptr] = cx - 1; stack_ptr += 1
                    if cx < w - 1 and binary_img[cy, cx + 1] == 0 and not visited[cy, cx + 1]:
                        visited[cy, cx + 1] = True; stack_y[stack_ptr] = cy; stack_x[stack_ptr] = cx + 1; stack_ptr += 1
                
                if not touches_border and current_count > max_area:
                    max_area = current_count
                    best_pixels_y = curr_comp_y
                    best_pixels_x = curr_comp_x
                    
    final_pixels = []
    for i in range(len(best_pixels_y)):
        final_pixels.append((best_pixels_y[i] + roi_offset_y, best_pixels_x[i] + roi_offset_x))
    return final_pixels



def threshold_image(frame):
    gray = 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
    gray = gray.astype(np.uint8)
    thresh_val = 150
    binary = np.zeros_like(gray)
    binary[gray >= thresh_val] = 255
    return binary


def split_ROI(binary_image, min_sheet_area=500, scale=8):
    from collections import deque
    h, w = binary_image.shape
    h_small, w_small = h // scale, w // scale
    downsampled = binary_image[::scale, ::scale]
    visited = np.zeros((h_small, w_small), dtype=np.bool_)
    islands = []
    min_area_small = max(1, min_sheet_area // (scale * scale))
    white_pixel_coords = np.argwhere(downsampled == 255) 
    for start_y, start_x in white_pixel_coords:
        if visited[start_y, start_x]: continue
        island = []
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            island.append((y * scale, x * scale))
            if y > 0 and downsampled[y-1, x] == 255 and not visited[y-1, x]:
                visited[y-1, x] = True; queue.append((y-1, x))
            if y < h_small-1 and downsampled[y+1, x] == 255 and not visited[y+1, x]:
                visited[y+1, x] = True; queue.append((y+1, x))
            if x > 0 and downsampled[y, x-1] == 255 and not visited[y, x-1]:
                visited[y, x-1] = True; queue.append((y, x-1))
            if x < w_small-1 and downsampled[y, x+1] == 255 and not visited[y, x+1]:
                visited[y, x+1] = True; queue.append((y, x+1))
        if len(island) >= min_area_small:
            islands.append(island)
    return islands

def detect_tag(image, island, gray_image, scale=8):
    h, w = gray_image.shape
    island_arr = np.array(island, dtype=np.int32)
    min_y, max_y = island_arr[:, 0].min(), island_arr[:, 0].max()
    min_x, max_x = island_arr[:, 1].min(), island_arr[:, 1].max()
    for _ in range(scale + 2):
        if min_y > 0 and np.any(gray_image[min_y - 1, min_x : max_x + 1] == 255): min_y -= 1
        else: break
    for _ in range(scale + 2):
        if max_y < h - 1 and np.any(gray_image[max_y + 1, min_x : max_x + 1] == 255): max_y += 1
        else: break
    for _ in range(scale + 2):
        if min_x > 0 and np.any(gray_image[min_y : max_y + 1, min_x - 1] == 255): min_x -= 1
        else: break
    for _ in range(scale + 2):
        if max_x < w - 1 and np.any(gray_image[min_y : max_y + 1, max_x + 1] == 255): max_x += 1
        else: break
    pad = 1
    safe_min_y = max(0, min_y - pad)
    safe_max_y = min(h, max_y + 1 + pad)
    safe_min_x = max(0, min_x - pad)
    safe_max_x = min(w, max_x + 1 + pad)
    
    roi = gray_image[safe_min_y : safe_max_y, safe_min_x : safe_max_x]
    tag_pixels = fast_component_labeling_local(roi, safe_min_y, safe_min_x)
    if not tag_pixels:
        return []
    roi_h, roi_w = roi.shape
    mask = np.zeros((roi_h, roi_w), dtype=np.bool_)
    local_y = [p[0] - safe_min_y for p in tag_pixels]
    local_x = [p[1] - safe_min_x for p in tag_pixels]
    mask[local_y, local_x] = True
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    eroded = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & 
              padded[1:-1, :-2] & padded[1:-1, 2:])
    boundary_mask = mask ^ eroded
    by, bx = np.where(boundary_mask)
    marker_pixels = []
    for i in range(len(by)):
        marker_pixels.append((by[i] + safe_min_y, bx[i] + safe_min_x))
    return marker_pixels

def get_corners(marker_pixels):
    if not marker_pixels or len(marker_pixels) < 4: return None
    pts = np.array([[p[1], p[0]] for p in marker_pixels], dtype=np.float32)
    hull = convex_hull(pts) # From manual_cv2
    perimeter = arc_length(hull, True)
    corners = None

    for eps_factor in np.linspace(0.01, 0.08, 8):
        approx = approx_poly_dp(hull, eps_factor * perimeter, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            break

    if corners is None: return None
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    idx = np.lexsort((corners[:, 0], corners[:, 1]))[0]
    corners = np.roll(corners, -idx, axis=0)
    return corners.astype(np.float32)

class ARtag:
    def __init__(self, corners, id, orientation_steps):
        self.corners = corners
        self.id = id
        self.orientation_steps = orientation_steps
        self.rotation = orientation_steps * 90

def compute_homography_manual(src_pts, dst_pts):
    """Solves Homography using SVD (No OpenCV)"""
    A = []
    for i in range(4):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def perspective_transform_manual(points, matrix):
    """Project points using matrix (No OpenCV)"""
    pts = points.reshape(-1, 3)
    projected = matrix @ pts.T # (3, N)
    projected = projected.T # (N, 3)
    z = projected[:, 2:3]
    z[np.abs(z) < 1e-9] = 1.0 # Avoid div zero
    out = projected[:, :2] / z
    return out

def decode_tag(marker_pixels, gray_image): 
    corners = get_corners(marker_pixels)
    if corners is None: return None
    src_pts = np.array([[0,0],[8,0],[8,8],[0,8]], dtype=np.float32)
    dst_pts = corners.astype(np.float32)
    H = compute_homography_manual(src_pts, dst_pts)
    sample_resolution = 5
    margin = 0.2 
    offsets = np.linspace(-0.5 + margin, 0.5 - margin, sample_resolution)
    grid_vals = np.zeros((8, 8), dtype=np.uint8)
    h, w = gray_image.shape

    for i in range(8):       # Columns
        for j in range(8):   # Rows
            cell_samples = []
            for dy in offsets:
                for dx in offsets:
                    u, v = i + 0.5 + dx, j + 0.5 + dy
                    
                    denom = H[2, 0]*u + H[2, 1]*v + H[2, 2]
                    if abs(denom) < 1e-6: continue 
                    
                    x_img = (H[0, 0]*u + H[0, 1]*v + H[0, 2]) / denom
                    y_img = (H[1, 0]*u + H[1, 1]*v + H[1, 2]) / denom
                    
                    if 0 <= int(y_img) < h and 0 <= int(x_img) < w:
                        cell_samples.append(gray_image[int(y_img), int(x_img)])

            # MEDIAN VOTE (robust to noise)
            if len(cell_samples) > 0:
                cell_samples.sort()
                median_pixel = cell_samples[len(cell_samples) // 2]
                grid_vals[j, i] = 1 if median_pixel > 127 else 0

    
    if grid_vals[5, 5] == 1: rotation = 0
    elif grid_vals[5, 2] == 1: rotation = 1
    elif grid_vals[2, 2] == 1: rotation = 2
    elif grid_vals[2, 5] == 1: rotation = 3
    else: return None
    
    if rotation > 0:
        grid_vals = np.rot90(grid_vals, k=rotation)
        corners = np.roll(corners, shift=-rotation, axis=0)

    found_id = (grid_vals[3,3]*8) + (grid_vals[3,4]*4) + (grid_vals[4,4]*2) + (grid_vals[4,3]*1)
    return ARtag(corners, found_id, rotation)

def compute_projection_matrix(corners, K):
    src_pts = np.array([[0,0], [200,0], [200,200], [0,200]], dtype=np.float32)
    H = compute_homography_manual(src_pts, corners)
    try:
        inv_K = np.linalg.inv(K)
        A = inv_K @ H
        col1 = A[:, 0]; col2 = A[:, 1]; col3 = A[:, 2]
        norm = (np.linalg.norm(col1) + np.linalg.norm(col2)) / 2.0
        if norm == 0: return None
        r1 = col1 / norm; r2 = col2 / norm; t = col3 / norm
        r3 = np.cross(r1, r2)
        R = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        extrinsics = np.column_stack([R, t])
        P = K @ extrinsics
        return P
    except: return None


class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.faces = []
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz: v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'f':
                face = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                self.faces.append(face)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, scale=3, color=False):
    """
    Render a loaded obj model into the current video frame.

    Args:
        img: The current video frame.
        obj: The loaded OBJ model.
        projection: The 3D projection matrix.
        scale: Scale factor for the 3D model. Defaults to 3.
        color: Whether to render in color. Defaults to False.
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    h, w = 200, 200  # AR tag reference size

    for face in obj.faces:
        face_vertices = face  # face is already a list of vertex indices
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)

    return img

def read_intrinsics(path):
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
    try:
        with open(path, 'r') as f:
            content = f.read()
            import re
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
            if len(nums) >= 9:
                K = np.array(nums[:9], dtype=np.float32).reshape(3, 3)
    except: pass
    return K


_obj_cache = {}

def process_frame_3D(frame, model_path, intrinsics_path, scale_3d=50.0, scale=4):
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    K = read_intrinsics(intrinsics_path)
    if model_path not in _obj_cache:
        _obj_cache[model_path] = OBJ(model_path, swapyz=True)
    obj = _obj_cache[model_path]
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                P = compute_projection_matrix(tag.corners, K)
                if P is not None:
                    frame = render(frame, obj, P, scale=scale_3d)
    return frame

def process_frame_marking(frame, scale=4):
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    detected_tags = []
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                detected_tags.append(tag)
    
    for tag in detected_tags:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        for i, point in enumerate(tag.corners):
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 8, colors[i], -1)
            next_point = tag.corners[(i + 1) % 4]
            nx, ny = int(next_point[0]), int(next_point[1])
            cv2.line(frame, (x,y), (nx,ny), (0,255,0), 2)
        
        cx = int(sum(c[0] for c in tag.corners) / 4)
        cy = int(sum(c[1] for c in tag.corners) / 4)
        text = f"ID:{tag.id} {tag.rotation}deg"
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    return frame

def process_frame_superimpose(frame, template_path, scale=4):
    template = cv2.imread(template_path)
    if template is None: return frame
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    h_temp, w_temp = template.shape[:2]
    src_pts = np.array([[0,0], [w_temp,0], [w_temp,h_temp], [0,h_temp]], dtype=np.float32)
    
    frame_h, frame_w = frame.shape[:2]
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                H = compute_homography_manual(src_pts, tag.corners)
                warped = warp_perspective(template, H, (frame_w, frame_h))
                mask_src = np.ones((h_temp, w_temp, 1), dtype=np.uint8) * 255
                mask_warped = warp_perspective(mask_src, H, (frame_w, frame_h))
                mask_bool = mask_warped[:, :, 0] > 0
                frame[mask_bool] = warped[mask_bool]
                
    return frame

