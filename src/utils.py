import cv2
import numpy as np
import math
from manual_cv2 import *


def generate_tag(cell_size=50, tag_id=0):
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[2, 2] = 0
    grid[2, 3] = 255
    grid[2, 4] = 255
    grid[2, 5] = 0
    grid[3, 2] = 255
    grid[3, 3] = 255  
    grid[3, 4] = 0  
    grid[3, 5] = 255
    
    # Row 4
    grid[4, 2] = 255
    grid[4, 3] = 255  
    grid[4, 4] = 255  
    grid[4, 5] = 255
    
    # Row 5
    grid[5, 2] = 255
    grid[5, 3] = 255
    grid[5, 4] = 255
    grid[5, 5] = 0

    # Scale the 8x8 grid to a visible image size
    tag_image = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    
    cv2.imwrite(f"Tag{tag_id}.png", tag_image)

    return tag_image

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame.

    Args:
        img: The current video frame.
        obj: The loaded OBJ model.
        projection: The 3D projection matrix.
        model: The reference image representing the surface to be augmented.
        color: Whether to render in color. Defaults to False.
    """
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = perspective_transform_points(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst[:, 0, :2])
        if color is False:
            fill_convex_poly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            fill_convex_poly(img, imgpts, color)

    return img


class ARtag:
    def __init__(self, corners, id):
        self.corners = corners
        self.id = id
        
    def code_to_id(self, code):
        id = 0
        for i in range(4):
            id += code[i] * (2 ** (3 - i));
        return id
    


def threshold_image(frame):
    """Convert the image so that only the black and white regions are visible, and rest are masked

    Args:
        frame (): It is a numpy array of shape (H, W, 3) representing the input image
        Returns:
        bin_frame: A binary image of shape (H, W) where the pixels are either 0 or 255. The black and white regions are represented by 255, and the rest are represented by 0.
    """
    gray = 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
    # threshold the image, without using cv2
    thresh = 150
    gray[gray < thresh] = 0
    gray[gray >= thresh] = 255
    
    return gray.astype(np.uint8)

def split_ROI(binary_image, min_sheet_area=500):
    """Get the connected components in the binary image
    
    Args:
        binary_image: A binary image of shape (H, W) where the pixels are either 0 or 255. The white regions are represented by 255, and the black regions are represented by 0.
        Returns:
        islands: A list of islands, where each island is a list of (y, x) coordinates of the pixels belonging to that island. Only islands with area greater than or equal to min_sheet_area are returned.
    """
    h, w = binary_image.shape
    visited = np.zeros((h, w), dtype=bool)
    islands = []
    
    x_mov = [-1, 0, 1, 0]
    y_mov = [0, 1, 0, -1]
    for y in range(h):
        for x in range(w):
            if binary_image[y, x] == 255 and not visited[y, x]:
                island = []
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    island.append((cy, cx))
                    for direction in range(4):
                        ny, nx = cy + y_mov[direction], cx + x_mov[direction]
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary_image[ny, nx] == 255 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                if len(island) >= min_sheet_area:
                    islands.append(island)
                    
    return islands



def render_ROI(image, island, index):
    h, w, _ = image.shape
    new_image = np.zeros((h, w, 3), dtype=np.uint8)
    for (y, x) in island[index]:
        new_image[y, x] = image[y, x]
        
    return new_image


def detect_tag(image, island, gray_image):
    detected_tags = []
    h, w = gray_image.shape
    ys = [y for (y, x) in island]
    xs = [x for (y, x) in island]
    min_y, max_y = min(ys), max(ys)
    min_x, max_x = min(xs), max(xs)
    visited_black = set()
    marker_pixels = []
    x_hor = [0, 1, 0, -1]
    y_hor = [1, 0, -1, 0]
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if gray_image[y, x] == 0 and (y, x) not in visited_black:
                touching_border = False
                stack = [(y,x)]
                visited_black.add((y,x))
                black_region = []
                while stack:
                    cy, cx = stack.pop()
                    black_region.append((cy, cx))
                    for direction in range(4):
                        ny, nx = cy + y_hor[direction], cx + x_hor[direction]
                        if ny <= min_y or ny >= max_y or nx <= min_x or nx >= max_x:
                            touching_border = True
                            continue
                        
                        
                        if gray_image[ny, nx] == 0 and (ny, nx) not in visited_black:
                            visited_black.add((ny, nx))
                            stack.append((ny, nx))
                    # But how is it removing the corner black regions
                if not touching_border:
                    if len(black_region) > len(marker_pixels):
                        marker_pixels = black_region
    # Now we have the marker pixels, we can find the corners of the tag using the marker pixels, we can use the fact that the corners of the tag will be the farthest points
    
    return marker_pixels
                    


def get_corners(marker_pixels):
    if not marker_pixels:
        return None

    # 1. Convert to contour format (x, y)
    contour = np.array([[p[1], p[0]] for p in marker_pixels],
                       dtype=np.float32).reshape(-1, 1, 2)

    # 2. Convex Hull
    hull = convex_hull_contour(contour)

    # 3. Adaptive polygon approximation
    perimeter = arc_length(hull, True)
    corners = None

    for eps_factor in np.linspace(0.01, 0.08, 8):
        approx = approx_poly_dp(hull, eps_factor * perimeter, True)

        if len(approx) == 4 and is_contour_convex(approx):
            corners = approx.reshape(4, 2)
            break

    if corners is None:
        return None   # no fallback — better to reject than hallucinate

    # 4. Robust corner ordering (angle + anchor)
    center = np.mean(corners, axis=0)

    angles = np.arctan2(corners[:, 1] - center[1],
                        corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]

    # Anchor: top-left = smallest y, then x
    idx = np.lexsort((corners[:, 0], corners[:, 1]))[0]
    corners = np.roll(corners, -idx, axis=0)

    return corners.astype(np.float32)



def compute_homography(src_pts, dst_pts):
    A = []
    for i in range(4):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H


def decode_tag(marker_pixels, gray_image): 
    h, w = gray_image.shape
    
    # 1. Get Corners
    corners = get_corners(marker_pixels)
    if corners is None: return None

    # 2. Homography
    src_pts = np.array([[0,0],[8,0],[8,8],[0,8]], dtype=np.float32)
    dst_pts = corners.astype(np.float32)
    H = compute_homography(src_pts, dst_pts)
    
    # 3. Sample with Median Voting
    # Configuration: 5x5 grid, inner 60% of cell
    sample_resolution = 5
    margin = 0.2 
    offsets = np.linspace(-0.5 + margin, 0.5 - margin, sample_resolution)

    grid_vals = np.zeros((8, 8), dtype=np.uint8)

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

            # MEDIAN VOTE
            if len(cell_samples) > 0:
                cell_samples.sort()
                median_pixel = cell_samples[len(cell_samples) // 2]
                grid_vals[j, i] = 1 if median_pixel > 127 else 0

    # 4. Check Orientation
    rotation_steps = 0
    if grid_vals[5, 5] == 1: rotation_steps = 0
    elif grid_vals[5, 2] == 1: rotation_steps = 1
    elif grid_vals[2, 2] == 1: rotation_steps = 2
    elif grid_vals[2, 5] == 1: rotation_steps = 3
    else: return None

    # 5. Correct Orientation
    if rotation_steps > 0:
        grid_vals = np.rot90(grid_vals, k=rotation_steps)
        corners = np.roll(corners, shift=-rotation_steps, axis=0)

    # 6. Relaxed Border Validation (Allows 2 errors)
    border_errors = 0
    for k in range(8):
        if grid_vals[0, k] == 1: border_errors += 1
        if grid_vals[7, k] == 1: border_errors += 1
    for k in range(1, 7):
        if grid_vals[k, 0] == 1: border_errors += 1
        if grid_vals[k, 7] == 1: border_errors += 1
        
    if border_errors > 2:
        return None
    
    # 7. Extract ID
    found_id = (grid_vals[3,3]*8) + (grid_vals[3,4]*4) + (grid_vals[4,4]*2) + (grid_vals[4,3]*1)
    
    return ARtag(corners, found_id)



def mark_corners(image, tag):
    if tag is None:
        return
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    corners = tag.corners
    for i, point in enumerate(corners):
        x, y = int(point[0]), int(point[1])
        draw_circle(image, (x, y), 8, colors[i], -1)
        # Draw line to next corner
        next_point = corners[(i + 1) % 4]
        next_x, next_y = int(next_point[0]), int(next_point[1])
        draw_line(image, (x, y), (next_x, next_y), (0, 255, 0), 2)
    
    cx = int(sum(c[0] for c in corners) / 4)
    cy = int(sum(c[1] for c in corners) / 4)
    put_text(image, str(tag.id), (cx, cy), FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, LINE_AA)




# def superimpose(video_frame, template_image, tag):
#     if tag is None:
#         return video_frame
#     template_height, template_width = template_image.shape[:2]
#     src_corners = np.array([
#         [0, 0],                                      # Top-left
#         [template_width - 1, 0],                     # Top-right
#         [template_width - 1, template_height - 1],   # Bottom-right
#         [0, template_height - 1]                     # Bottom-left
#     ], dtype=np.float32)
#     dst_corners = tag.corners.astype(np.float32)
#     H = compute_homography(src_corners, dst_corners)
#     frame_height, frame_width = video_frame.shape[:2]
#     warped_template = warp_perspective(
#         template_image, 
#         H, 
#         (frame_width, frame_height),
#     )
#     gray_warped = rgb_to_grayscale(warped_template)
#     mask = threshold_binary(gray_warped, 1, 255)
#     mask_inv = bitwise_not(mask)
#     video_background = bitwise_and_with_mask(video_frame, mask_inv)
#     template_foreground = bitwise_and_with_mask(warped_template, mask)
#     result = add_images(video_background, template_foreground)
#     return result


# def superimpose(video_frame, template_image, tag):
#     if tag is None:
#         return video_frame
#     h_temp, w_temp, _ = template_image.shape
#     src_pts = np.array([
#         [0, 0], 
#         [w_temp - 1, 0], 
#         [w_temp - 1, h_temp - 1], 
#         [0, h_temp - 1]
#     ], dtype=np.float32)
    
#     dst_pts = tag.corners.astype(np.float32)
#     H = compute_homography(src_pts, dst_pts)
#     frame_h, frame_w = video_frame.shape[:2]
#     cv2.warpPerspective(template_image, H, (frame_w, frame_h), dst=video_frame, borderMode=cv2.BORDER_TRANSPARENT)
#     return video_frame


# def get_warp_perspective(
#     src_image,
#     h_matrix,
#     out_shape,
#     roi=None,
#     interpolation="bilinear",
#     warp_inverse_map=False,
#     border_mode="constant",
#     border_value=0,
# ):
#     """Warp src_image onto an output canvas using a homography.

#     This implements *backward warping* (iterate output pixels and sample from source).

#     Args:
#         src_image: Source image (H, W) or (H, W, C).
#         h_matrix: Homography mapping source (x, y, 1) -> destination (x', y', 1).
#         out_shape: (out_h, out_w) of the destination canvas.

#     Returns:
#         warped image with shape (out_h, out_w) or (out_h, out_w, C).
#     """
#     out_h, out_w = int(out_shape[0]), int(out_shape[1])
#     src_h, src_w = src_image.shape[:2]

#     # Fast path: native module (if built) for uint8 images.
#     # Note: only supports roi=None (callers can warp ROI-sized canvases instead).
#     if (
#         _ar_native is not None
#         and roi is None
#         and isinstance(src_image, np.ndarray)
#         and src_image.dtype == np.uint8
#         and (src_image.ndim == 2 or src_image.ndim == 3)
#     ):
#         interp = 1 if interpolation == "bilinear" else 0
#         bmode = 1 if border_mode == "replicate" else 0
#         return _ar_native.warp_perspective_u8(
#             src_image,
#             np.asarray(h_matrix, dtype=np.float64),
#             out_h,
#             out_w,
#             interp,
#             bool(warp_inverse_map),
#             bmode,
#             border_value,
#         )

#     if src_image.ndim == 2:
#         warped = np.zeros((out_h, out_w), dtype=src_image.dtype)
#     else:
#         warped = np.zeros((out_h, out_w, src_image.shape[2]), dtype=src_image.dtype)

#     # Backward map: dst -> src. Optionally restrict work to a ROI.
#     # roi is (x0, y0, x1, y1) inclusive bounds in destination coords.
#     if roi is None:
#         x0, y0, x1, y1 = 0, 0, out_w - 1, out_h - 1
#     else:
#         x0, y0, x1, y1 = roi
#         x0 = max(0, min(out_w - 1, int(x0)))
#         x1 = max(0, min(out_w - 1, int(x1)))
#         y0 = max(0, min(out_h - 1, int(y0)))
#         y1 = max(0, min(out_h - 1, int(y1)))
#         if x1 < x0 or y1 < y0:
#             return warped

#     # OpenCV semantics:
#     # - Internally uses a matrix that maps dst -> src.
#     # - If WARP_INVERSE_MAP is not set, OpenCV inverts the provided matrix.
#     if warp_inverse_map:
#         M = np.array(h_matrix, dtype=np.float64)
#     else:
#         M = np.linalg.inv(h_matrix).astype(np.float64)

#     if border_mode not in {"constant", "replicate"}:
#         raise ValueError("border_mode must be 'constant' or 'replicate'")

#     if src_image.ndim == 2:
#         border_val = float(border_value)
#     else:
#         if isinstance(border_value, (tuple, list, np.ndarray)):
#             if len(border_value) != src_image.shape[2]:
#                 raise ValueError("border_value length must match channel count")
#             border_val = np.array(border_value, dtype=np.float64)
#         else:
#             border_val = np.full((src_image.shape[2],), float(border_value), dtype=np.float64)

#     h00, h01, h02 = M[0, 0], M[0, 1], M[0, 2]
#     h10, h11, h12 = M[1, 0], M[1, 1], M[1, 2]
#     h20, h21, h22 = M[2, 0], M[2, 1], M[2, 2]

#     use_bilinear = (interpolation == "bilinear")

#     for y in range(y0, y1 + 1):
#         for x in range(x0, x1 + 1):
#             denom = (h20 * x) + (h21 * y) + h22
#             if abs(denom) < 1e-9:
#                 continue
#             sx = ((h00 * x) + (h01 * y) + h02) / denom
#             sy = ((h10 * x) + (h11 * y) + h12) / denom

#             if use_bilinear:
#                 # Bilinear interpolation (similar to cv2.INTER_LINEAR)
#                 ix = int(np.floor(sx))
#                 iy = int(np.floor(sy))
#                 dx = sx - ix
#                 dy = sy - iy

#                 # Fast path if fully inside image (no border needed)
#                 if 0 <= ix < (src_w - 1) and 0 <= iy < (src_h - 1):
#                     if src_image.ndim == 2:
#                         p00 = float(src_image[iy, ix])
#                         p10 = float(src_image[iy, ix + 1])
#                         p01 = float(src_image[iy + 1, ix])
#                         p11 = float(src_image[iy + 1, ix + 1])
#                         val = (
#                             (1 - dx) * (1 - dy) * p00
#                             + dx * (1 - dy) * p10
#                             + (1 - dx) * dy * p01
#                             + dx * dy * p11
#                         )
#                         if warped.dtype == np.uint8:
#                             warped[y, x] = np.uint8(np.clip(val + 0.5, 0, 255))
#                         else:
#                             warped[y, x] = val
#                     else:
#                         p00 = src_image[iy, ix].astype(np.float64)
#                         p10 = src_image[iy, ix + 1].astype(np.float64)
#                         p01 = src_image[iy + 1, ix].astype(np.float64)
#                         p11 = src_image[iy + 1, ix + 1].astype(np.float64)
#                         val = (
#                             (1 - dx) * (1 - dy) * p00
#                             + dx * (1 - dy) * p10
#                             + (1 - dx) * dy * p01
#                             + dx * dy * p11
#                         )
#                         if warped.dtype == np.uint8:
#                             warped[y, x] = np.uint8(np.clip(val + 0.5, 0, 255))
#                         else:
#                             warped[y, x] = val
#                     continue

#                 # Border-aware sampling
#                 x0s, x1s = ix, ix + 1
#                 y0s, y1s = iy, iy + 1

#                 def sample(px, py):
#                     if border_mode == "replicate":
#                         px = 0 if px < 0 else (src_w - 1 if px >= src_w else px)
#                         py = 0 if py < 0 else (src_h - 1 if py >= src_h else py)
#                         return src_image[py, px]
#                     # constant
#                     if px < 0 or py < 0 or px >= src_w or py >= src_h:
#                         return border_val
#                     return src_image[py, px]

#                 if src_image.ndim == 2:
#                     p00 = float(sample(x0s, y0s))
#                     p10 = float(sample(x1s, y0s))
#                     p01 = float(sample(x0s, y1s))
#                     p11 = float(sample(x1s, y1s))
#                     val = (
#                         (1 - dx) * (1 - dy) * p00
#                         + dx * (1 - dy) * p10
#                         + (1 - dx) * dy * p01
#                         + dx * dy * p11
#                     )
#                     if warped.dtype == np.uint8:
#                         warped[y, x] = np.uint8(np.clip(val + 0.5, 0, 255))
#                     else:
#                         warped[y, x] = val
#                 else:
#                     p00 = sample(x0s, y0s)
#                     p10 = sample(x1s, y0s)
#                     p01 = sample(x0s, y1s)
#                     p11 = sample(x1s, y1s)
#                     p00 = p00.astype(np.float64) if isinstance(p00, np.ndarray) else np.array(p00, dtype=np.float64)
#                     p10 = p10.astype(np.float64) if isinstance(p10, np.ndarray) else np.array(p10, dtype=np.float64)
#                     p01 = p01.astype(np.float64) if isinstance(p01, np.ndarray) else np.array(p01, dtype=np.float64)
#                     p11 = p11.astype(np.float64) if isinstance(p11, np.ndarray) else np.array(p11, dtype=np.float64)
#                     val = (
#                         (1 - dx) * (1 - dy) * p00
#                         + dx * (1 - dy) * p10
#                         + (1 - dx) * dy * p01
#                         + dx * dy * p11
#                     )
#                     if warped.dtype == np.uint8:
#                         warped[y, x] = np.uint8(np.clip(val + 0.5, 0, 255))
#                     else:
#                         warped[y, x] = val
#             else:
#                 # Nearest neighbor
#                 ix = int(sx)
#                 iy = int(sy)
#                 if border_mode == "replicate":
#                     ix = 0 if ix < 0 else (src_w - 1 if ix >= src_w else ix)
#                     iy = 0 if iy < 0 else (src_h - 1 if iy >= src_h else iy)
#                     warped[y, x] = src_image[iy, ix]
#                 else:
#                     if 0 <= ix < src_w and 0 <= iy < src_h:
#                         warped[y, x] = src_image[iy, ix]
#                     else:
#                         warped[y, x] = border_val

#     return warped


def get_warp_perspective(
    src_image,
    h_matrix,
    out_shape,
    roi=None,
    interpolation="bilinear",
    warp_inverse_map=False,
    border_mode="constant",
    border_value=0,
):
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    src_h, src_w = src_image.shape[:2]

    # 1. Prepare ROI
    if roi is None:
        x0, y0, x1, y1 = 0, 0, out_w - 1, out_h - 1
    else:
        x0, y0, x1, y1 = roi
        x0, x1 = max(0, int(x0)), min(out_w - 1, int(x1))
        y0, y1 = max(0, int(y0)), min(out_h - 1, int(y1))

    # 2. Handle Matrix Inversion (Backward Mapping)
    M = np.array(h_matrix, dtype=np.float64)
    if not warp_inverse_map:
        M = np.linalg.inv(M)

    # 3. Create a grid of destination coordinates for the ROI
    yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
    targets = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()])

    # 4. Transform all coordinates at once
    transformed = M @ targets
    sx = transformed[0] / (transformed[2] + 1e-9)
    sy = transformed[1] / (transformed[2] + 1e-9)

    # 5. Interpolation
    warped_roi_shape = (y1 - y0 + 1, x1 - x0 + 1)
    if interpolation == "nearest":
        ix, iy = np.round(sx).astype(int), np.round(sy).astype(int)
    else: # Bilinear
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        dx, dy = sx - ix, sy - iy

    # 6. Mask for valid source pixels
    mask = (sx >= 0) & (sx < src_w - 1) & (sy >= 0) & (sy < src_h - 1)
    
    # Initialize output canvas
    channels = src_image.shape[2] if src_image.ndim == 3 else 1
    out_img_shape = (out_h, out_w, channels) if channels > 1 else (out_h, out_w)
    warped = np.full(out_img_shape, border_value, dtype=src_image.dtype)

    # Apply valid pixels
    if interpolation == "bilinear":
        # Get 4 neighboring pixels for all points at once
        v00 = src_image[iy[mask], ix[mask]]
        v10 = src_image[iy[mask], ix[mask] + 1]
        v01 = src_image[iy[mask] + 1, ix[mask]]
        v11 = src_image[iy[mask] + 1, ix[mask] + 1]

        # Weighted sum (Vectorized)
        w00 = (1 - dx[mask]) * (1 - dy[mask])
        w10 = dx[mask] * (1 - dy[mask])
        w01 = (1 - dx[mask]) * dy[mask]
        w11 = dx[mask] * dy[mask]

        # Reshape weights if color image
        if channels > 1:
            w00, w10, w01, w11 = [w[:, None] for w in [w00, w10, w01, w11]]

        val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
        
        # Place ROI back into full canvas
        roi_view = warped[y0:y1+1, x0:x1+1]
        roi_view.reshape(-1, channels)[mask] = val.astype(src_image.dtype)
    else:
        # Nearest Neighbor logic (simpler)
        valid_mask = (ix >= 0) & (ix < src_w) & (iy >= 0) & (iy < src_h)
        roi_view = warped[y0:y1+1, x0:x1+1]
        roi_view.reshape(-1, channels)[valid_mask] = src_image[iy[valid_mask], ix[valid_mask]]

    return warped


def superimpose(video_frame, template_image, tag):
    if tag is None:
        return video_frame
    h_temp, w_temp, _ = template_image.shape
    src_pts = np.array([
        [0, 0], 
        [w_temp - 1, 0], 
        [w_temp - 1, h_temp - 1], 
        [0, h_temp - 1]
    ], dtype=np.float32)
    
    dst_pts = tag.corners.astype(np.float32)
    H = compute_homography(src_pts, dst_pts)

    frame_h, frame_w = video_frame.shape[:2]

    # Work only in the tag's bounding box.
    x_min = int(np.floor(dst_pts[:, 0].min()))
    x_max = int(np.ceil(dst_pts[:, 0].max()))
    y_min = int(np.floor(dst_pts[:, 1].min()))
    y_max = int(np.ceil(dst_pts[:, 1].max()))
    pad = 2
    x0 = max(0, x_min - pad)
    x1 = min(frame_w - 1, x_max + pad)
    y0 = max(0, y_min - pad)
    y1 = min(frame_h - 1, y_max + pad)

    roi_w = (x1 - x0 + 1)
    roi_h = (y1 - y0 + 1)

    # Shift destination coords into ROI coordinates so we can warp into a small canvas.
    T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_roi = T @ H

    warped_roi = get_warp_perspective(
        template_image,
        H_roi,
        (roi_h, roi_w),
        # roi=None,
        interpolation="bilinear",
    )

    # Mask in ROI coordinates
    dst_pts_roi = dst_pts.copy()
    dst_pts_roi[:, 0] -= x0
    dst_pts_roi[:, 1] -= y0
    mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
    fill_convex_poly(mask_roi, np.int32(dst_pts_roi), 255)
    mask_bool = mask_roi > 0

    out = video_frame.copy()
    out_slice = out[y0:y1 + 1, x0:x1 + 1]
    out_slice[mask_bool] = warped_roi[mask_bool]
    out[y0:y1 + 1, x0:x1 + 1] = out_slice

    # if DEBUG_WARP_COMPARE:
    #     if not hasattr(superimpose, "_frame_idx"):
    #         superimpose._frame_idx = 0
    #     superimpose._frame_idx += 1

    #     if superimpose._frame_idx % DEBUG_WARP_EVERY_N_FRAMES == 0:
    #         warp_cv_roi = cv2.warpPerspective(
    #             template_image,
    #             H_roi,
    #             (roi_w, roi_h),
    #             flags=cv2.INTER_LINEAR,
    #             borderMode=cv2.BORDER_CONSTANT,
    #             borderValue=0,
    #         )
    #         diff = cv2.absdiff(warp_cv_roi, warped_roi)
    #         err = float(np.mean(diff[mask_bool])) if np.any(mask_bool) else 0.0
    #         print(f"[warp debug] mean abs diff (masked ROI): {err:.3f}")
    #         cv2.imshow("warp_cv2_roi", warp_cv_roi)
    #         cv2.imshow("warp_manual_roi", warped_roi)
    #         cv2.imshow("warp_diff_roi", diff)

    return out

    

def process_frame(frame):
    # Cache template image; disk IO per frame is very slow.
    if not hasattr(process_frame, "_template_img"):
        process_frame._template_img = cv2.imread('./assets/iitd_logo_template.jpg')
    template_img = process_frame._template_img
    
    gray = threshold_image(frame)
    islands = split_ROI(gray)
    detected_tags = []
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            corners = get_corners(marker_pixels)
            if corners is not None:
                for i, (x, y) in enumerate(corners):
                    draw_circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
                    put_text(frame, str(i), (int(x) + 5, int(y) + 5),
                             FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, LINE_AA)
            if tag:
                detected_tags.append(tag)      
    for tag in detected_tags:
        # mark_corners(frame, tag)
        if template_img is not None:
            frame = superimpose(frame, template_img, tag)

    return frame


# -----------------------------------------------------------------------------
# Original cv2-based implementations (commented out per request)
# -----------------------------------------------------------------------------
# def render(img, obj, projection, model, color=False):
#     DEFAULT_COLOR = (0, 0, 0)
#     vertices = obj.vertices
#     scale_matrix = np.eye(3) * 3
#     h, w = model.shape
#
#     for face in obj.faces:
#         face_vertices = face[0]
#         points = np.array([vertices[vertex - 1] for vertex in face_vertices])
#         points = np.dot(points, scale_matrix)
#         points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
#         dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
#         imgpts = np.int32(dst)
#         if color is False:
#             cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
#         else:
#             color = hex_to_rgb(face[-1])
#             color = color[::-1]
#             cv2.fillConvexPoly(img, imgpts, color)
#
#     return img
#
#
# def get_corners(marker_pixels):
#     if not marker_pixels:
#         return None
#
#     contour = np.array([[p[1], p[0]] for p in marker_pixels],
#                        dtype=np.int32).reshape(-1, 1, 2)
#
#     hull = cv2.convexHull(contour)
#
#     perimeter = cv2.arcLength(hull, True)
#     corners = None
#
#     for eps_factor in np.linspace(0.01, 0.08, 8):
#         approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)
#
#         if len(approx) == 4 and cv2.isContourConvex(approx):
#             corners = approx.reshape(4, 2)
#             break
#
#     if corners is None:
#         return None
#
#     center = np.mean(corners, axis=0)
#     angles = np.arctan2(corners[:, 1] - center[1],
#                         corners[:, 0] - center[0])
#     corners = corners[np.argsort(angles)]
#     idx = np.lexsort((corners[:, 0], corners[:, 1]))[0]
#     corners = np.roll(corners, -idx, axis=0)
#
#     return corners.astype(np.float32)
#
#
# def mark_corners(image, tag):
#     if tag is None:
#         return
#     colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
#     corners = tag.corners
#     for i, point in enumerate(corners):
#         x, y = int(point[0]), int(point[1])
#         cv2.circle(image, (x, y), 8, colors[i], -1)
#         next_point = corners[(i + 1) % 4]
#         next_x, next_y = int(next_point[0]), int(next_point[1])
#         cv2.line(image, (x, y), (next_x, next_y), (0, 255, 0), 2)
#
#     cx = int(sum(c[0] for c in corners) / 4)
#     cy = int(sum(c[1] for c in corners) / 4)
#     cv2.putText(image, str(tag.id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#
# def superimpose(video_frame, template_image, tag):
#     if tag is None:
#         return video_frame
#     h_temp, w_temp, _ = template_image.shape
#     src_pts = np.array([
#         [0, 0],
#         [w_temp - 1, 0],
#         [w_temp - 1, h_temp - 1],
#         [0, h_temp - 1]
#     ], dtype=np.float32)
#
#     dst_pts = tag.corners.astype(np.float32)
#     H = compute_homography(src_pts, dst_pts)
#
#     frame_h, frame_w = video_frame.shape[:2]
#
#     x_min = int(np.floor(dst_pts[:, 0].min()))
#     x_max = int(np.ceil(dst_pts[:, 0].max()))
#     y_min = int(np.floor(dst_pts[:, 1].min()))
#     y_max = int(np.ceil(dst_pts[:, 1].max()))
#     pad = 2
#     x0 = max(0, x_min - pad)
#     x1 = min(frame_w - 1, x_max + pad)
#     y0 = max(0, y_min - pad)
#     y1 = min(frame_h - 1, y_max + pad)
#
#     roi_w = (x1 - x0 + 1)
#     roi_h = (y1 - y0 + 1)
#
#     T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]], dtype=np.float64)
#     H_roi = T @ H
#
#     warped_roi = get_warp_perspective(
#         template_image,
#         H_roi,
#         (roi_h, roi_w),
#         interpolation="bilinear",
#     )
#
#     dst_pts_roi = dst_pts.copy()
#     dst_pts_roi[:, 0] -= x0
#     dst_pts_roi[:, 1] -= y0
#     mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
#     cv2.fillConvexPoly(mask_roi, np.int32(dst_pts_roi), 255)
#     mask_bool = mask_roi > 0
#
#     out = video_frame.copy()
#     out_slice = out[y0:y1 + 1, x0:x1 + 1]
#     out_slice[mask_bool] = warped_roi[mask_bool]
#     out[y0:y1 + 1, x0:x1 + 1] = out_slice
#
#     return out
#
#
# def process_frame(frame):
#     if not hasattr(process_frame, "_template_img"):
#         process_frame._template_img = cv2.imread('./assets/iitd_logo_template.jpg')
#     template_img = process_frame._template_img
#
#     gray = threshold_image(frame)
#     islands = split_ROI(gray)
#     detected_tags = []
#
#     for island in islands:
#         marker_pixels = detect_tag(frame, island, gray)
#         if marker_pixels:
#             tag = decode_tag(marker_pixels, gray)
#             corners = get_corners(marker_pixels)
#             if corners is not None:
#                 for i, (x, y) in enumerate(corners):
#                     cv2.circle(frame, (int(x), int(y)), 6, (0,255,0), -1)
#                     cv2.putText(frame, str(i), (int(x)+5, int(y)+5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
#             if tag:
#                 detected_tags.append(tag)
#     for tag in detected_tags:
#         if template_img is not None:
#             frame = superimpose(frame, template_img, tag)
#
#     return frame



