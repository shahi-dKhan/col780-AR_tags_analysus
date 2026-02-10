import cv2
import numpy as np
import math

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
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

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
    thresh = 220
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
    # Can we first completely isolate the tag, like first we need to include the black part in the island, which is the outer border of the tag
    # But how are we supposed to mask the complete background, how do we differentiate between the black background, and the black border of the tag
    h, w = gray_image.shape
    # First, we can find the bounding box of the island, and then we can check if the bounding box contains a tag or not
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
                    
                
# def get_corners(marker_pixels):
#     print("Marker pixels:", np.array(marker_pixels).shape)
#     pts = np.array(marker_pixels)
#     # Use the fact that the corners will be the farthest points from each other
#     s = pts.sum(axis=1)
#     tl = pts[np.argmin(s)] # top-left has the smallest sum
#     br = pts[np.argmax(s)] # bottom-right has the largest sum
#     diff = np.diff(pts, axis=1)
#     tr = pts[np.argmin(diff)] # top-right has the smallest difference
#     bl = pts[np.argmax(diff)] # bottom-left has the largest difference
#     return np.array([
#         [tl[1], tl[0]], 
#         [tr[1], tr[0]], 
#         [br[1], br[0]], 
#         [bl[1], bl[0]]
#     ], dtype=np.float32)

def get_corners(marker_pixels):
    if not marker_pixels:
        return None
    
    pts = np.array(marker_pixels)
    # 1. Calculate the centroid of the black blob
    center = np.mean(pts, axis=0) # (y, x)
    
    # 2. Define search directions (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    # These represent the diagonal axes of a square
    diagonals = np.array([
        [-1, -1], # Towards Top-Left
        [-1,  1], # Towards Top-Right
        [ 1,  1], # Towards Bottom-Right
        [ 1, -1]  # Towards Bottom-Left
    ])
    
    corners = []
    for d in diagonals:
        # Subtract center to get vectors from centroid to every pixel
        vectors = pts - center
        # Dot product finds how far each pixel extends in direction 'd'
        projections = np.dot(vectors, d)
        # The pixel that reaches the furthest in that diagonal direction is the corner
        corners.append(pts[np.argmax(projections)])
        
    corners = np.array(corners)
    
    # Flip (y, x) to (x, y) for Homography matrix consistency
    return np.array([[p[1], p[0]] for p in corners], dtype=np.float32)

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

def decode_tag(marker_pixels, gray_image): # Fixed argument order
    h, w = gray_image.shape
    
    # 1. Get Geometric Corners
    corners = get_corners(marker_pixels)
    
    # 2. Compute Homography (Ideal 8x8 -> Image)
    src_pts = np.array([[0,0],[8,0],[8,8],[0,8]], dtype=np.float32)
    dst_pts = corners.astype(np.float32)
    H = compute_homography(src_pts, dst_pts)
    
    # 3. Sample the Grid
    # grid_vals = np.zeros((8,8), dtype=np.uint8)
    # offsets = [-0.25,0,0.25]
    # for i in range(8):
    #     for j in range(8):
    #         cell_samples = []
    #         for dx in offsets:
    #             for dy in offsets:
    #                 u_sample, v_sample = i + 0.5 + dx, j + 0.5 + dy
    #                 denom = H[2,0]*u_sample + H[2,1]*v_sample + H[2,2]
    #                 if abs(denom) < 1e-6:
    #                     continue
    #                 x_img = (H[0,0]*u_sample + H[0,1]*v_sample + H[0,2]) / denom
    #                 y_img = (H[1,0]*u_sample + H[1,1]*v_sample + H[1,2]) / denom
    #                 if 0 <= int(y_img) < h and 0 <= int(x_img) < w:
    #                     cell_samples.append(gray_image[int(y_img), int(x_img)])
    #         if cell_samples:
    #             if np.mean(cell_samples) > 127:
    #                 grid_vals[j, i] = 1
    #             else:
    #                 grid_vals[j, i] = 0
                    
                                                 
    #         # u, v = i + 0.5, j + 0.5 # Sample center of cell
            
    #         # denom = H[2,0]*u + H[2,1]*v + H[2,2]
    #         # if abs(denom) < 1e-6: continue
            
    #         # x_img = (H[0,0]*u + H[0,1]*v + H[0,2]) / denom
    #         # y_img = (H[1,0]*u + H[1,1]*v + H[1,2]) / denom
            
    #         # if 0 <= int(y_img) < h and 0 <= int(x_img) < w:
    #         #     # White = 1, Black = 0
    #         #     if gray_image[int(y_img), int(x_img)] > 127:
    #         #         grid_vals[j, i] = 1 # Store as (row, col)
    #         #     else:
    #         #         grid_vals[j, i] = 0
    # Configuration
    sample_resolution = 5  # 5x5 grid = 25 points per cell
    margin = 0.2           # 20% margin on each side (only sample inner 60%)
    
    # Pre-calculate offsets (e.g., from -0.3 to +0.3)
    # 0 is the center. 0.5 is the edge.
    start = -0.5 + margin
    end = 0.5 - margin
    offsets = np.linspace(start, end, sample_resolution)

    grid_vals = np.zeros((8, 8), dtype=np.uint8)

    for i in range(8):       # Grid Columns
        for j in range(8):   # Grid Rows
            
            cell_samples = []

            for dy in offsets:
                for dx in offsets:
                    # u, v are the ideal coordinates
                    u = i + 0.5 + dx
                    v = j + 0.5 + dy
                    
                    # Project ideal(u,v) -> image(x,y)
                    denom = H[2, 0]*u + H[2, 1]*v + H[2, 2]
                    if abs(denom) < 1e-6: continue 
                    
                    x_img = (H[0, 0]*u + H[0, 1]*v + H[0, 2]) / denom
                    y_img = (H[1, 0]*u + H[1, 1]*v + H[1, 2]) / denom
                    
                    if 0 <= int(y_img) < h and 0 <= int(x_img) < w:
                        cell_samples.append(gray_image[int(y_img), int(x_img)])

            # --- MEDIAN VOTING (Crucial for Glare) ---
            if len(cell_samples) > 0:
                cell_samples.sort()
                mid_point = len(cell_samples) // 2
                median_pixel = cell_samples[mid_point]
                
                # Threshold the Median
                grid_vals[j, i] = 1 if median_pixel > 127 else 0
    # 4. Check Orientation Marker
    # Locations: (5,5)=0deg, (5,2)=90deg CW, (2,2)=180deg, (2,5)=270deg CW
    rotation_steps = 0
    if grid_vals[5, 5] == 1:
        rotation_steps = 0
    elif grid_vals[5, 2] == 1: 
        rotation_steps = 1 # 90 deg Clockwise
    elif grid_vals[2, 2] == 1: 
        rotation_steps = 2 # 180 deg
    elif grid_vals[2, 5] == 1: 
        rotation_steps = 3 # 270 deg Clockwise
    else:
        return None # No marker found

    # 5. Correct Orientation
    if rotation_steps > 0:
        # grid_vals is rotated CW, so we rotate CCW to fix it.
        # np.rot90 rotates CCW by default. k=1 is 90 CCW.
        grid_vals = np.rot90(grid_vals, k=rotation_steps)
        
        # Corners must be shifted to match.
        # If image is 90 CW, the 'True TL' is currently at index 1 (TR).
        # We need to shift Left (negative roll) to move index 1 to index 0.
        corners = np.roll(corners, shift=-rotation_steps, axis=0)

    # 6. Validate Border (Must be Black)
    if grid_vals[0, 0] != 0 or grid_vals[0, 7] != 0 or \
       grid_vals[7, 0] != 0 or grid_vals[7, 7] != 0:
        return None
    
    # 7. Extract ID
    b1 = grid_vals[3, 3]
    b2 = grid_vals[3, 4]
    b3 = grid_vals[4, 4]
    b4 = grid_vals[4, 3]
    
    found_id = (b1 * 8) + (b2 * 4) + (b3 * 2) + (b4 * 1)
    
    return ARtag(corners, found_id)

def mark_corners(image, tag):
    if tag is None:
        return
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    corners = tag.corners
    for i, point in enumerate(corners):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 8, colors[i], -1)
        # Draw line to next corner
        next_point = corners[(i + 1) % 4]
        next_x, next_y = int(next_point[0]), int(next_point[1])
        cv2.line(image, (x, y), (next_x, next_y), (0, 255, 0), 2)
    
    cx = int(sum(c[0] for c in corners) / 4)
    cy = int(sum(c[1] for c in corners) / 4)
    cv2.putText(image, str(tag.id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def process_frame(frame):
    gray = threshold_image(frame)
    islands = split_ROI(gray)
    detected_tags = []
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray)
        if marker_pixels:
            tag = decode_tag(gray, marker_pixels)
            if tag:
                detected_tags.append(tag)
                
    # 3. Visualize all found tags
    for tag in detected_tags:
        mark_corners(frame, tag)
        # You can also call your render() function here
        # render(frame, obj, projection, tag, ...)

    return frame
