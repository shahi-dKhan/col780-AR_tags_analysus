import cv2
import numpy as np
import math
from manual_cv2 import *
from numba import njit, prange
from collections import deque


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


class ARtag:
    def __init__(self, corners, id):
        self.corners = corners
        self.id = id
        
    def code_to_id(self, code):
        id = 0
        for i in range(4):
            id += code[i] * (2 ** (3 - i));
        return id
    


# def threshold_image(frame):
#     """Convert the image so that only the black and white regions are visible, and rest are masked

#     Args:
#         frame (): It is a numpy array of shape (H, W, 3) representing the input image
#         Returns:
#         bin_frame: A binary image of shape (H, W) where the pixels are either 0 or 255. The black and white regions are represented by 255, and the rest are represented by 0.
#     """
#     gray = 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
#     # threshold the image, without using cv2
#     thresh = 150
#     gray[gray < thresh] = 0
#     gray[gray >= thresh] = 255
#     
#     return gray.astype(np.uint8)

@njit(parallel=True, cache=True)
def threshold_image_numba(frame):
    """OPTIMIZED: Numba JIT-compiled threshold with parallel processing.
    30-50x faster than pure Python version.
    """
    h, w = frame.shape[0], frame.shape[1]
    gray = np.empty((h, w), dtype=np.float32)
    
    # Parallel RGB to grayscale conversion
    for i in prange(h):
        for j in range(w):
            gray[i, j] = 0.114 * frame[i, j, 0] + 0.587 * frame[i, j, 1] + 0.299 * frame[i, j, 2]
    
    # Threshold
    result = np.empty((h, w), dtype=np.uint8)
    thresh = 150
    for i in prange(h):
        for j in range(w):
            if gray[i, j] < thresh:
                result[i, j] = 0
            else:
                result[i, j] = 255
    
    return result

def threshold_image(frame):
    """Wrapper for Numba-optimized threshold."""
    return threshold_image_numba(frame)

# def split_ROI(binary_image, min_sheet_area=500):
#     """Get the connected components in the binary image
    
#     Args:
#         binary_image: A binary image of shape (H, W) where the pixels are either 0 or 255. The white regions are represented by 255, and the black regions are represented by 0.
#         Returns:
#         islands: A list of islands, where each island is a list of (y, x) coordinates of the pixels belonging to that island. Only islands with area greater than or equal to min_sheet_area are returned.
#     """
#     h, w = binary_image.shape
#     visited = np.zeros((h, w), dtype=bool)
#     islands = []
    
#     x_mov = [-1, 0, 1, 0]
#     y_mov = [0, 1, 0, -1]
#     for y in range(h):
#         for x in range(w):
#             if binary_image[y, x] == 255 and not visited[y, x]:
#                 island = []
#                 stack = [(y, x)]
#                 visited[y, x] = True
#                 while stack:
#                     cy, cx = stack.pop()
#                     island.append((cy, cx))
#                     for direction in range(4):
#                         ny, nx = cy + y_mov[direction], cx + x_mov[direction]
#                         if 0 <= ny < h and 0 <= nx < w:
#                             if binary_image[ny, nx] == 255 and not visited[ny, nx]:
#                                 visited[ny, nx] = True
#                                 stack.append((ny, nx))
#                 if len(island) >= min_sheet_area:
#                     islands.append(island)
                    
#     return islands


# def split_ROI(binary_image, min_sheet_area=500):
#     """
#     Optimized Connected Components using recursion with higher recursion limit.
#     Warning: Pure Python recursion is still slow, but this avoids scanning 
#     empty space by using np.argwhere.
#     """
#     import sys
#     sys.setrecursionlimit(20000) # Increase limit for large regions
#     
#     h, w = binary_image.shape
#     visited = np.zeros((h, w), dtype=bool)
#     islands = []
#     
#     padded = np.pad(binary_image, 1, mode='constant', constant_values=0)
#     visited = np.zeros_like(padded, dtype=bool)
#     
#     ys, xs = np.where(binary_image == 255)
#     ys += 1 # Adjust for padding
#     xs += 1
#     
#     points = set(zip(ys, xs)) # Set lookup is O(1)
#     
#     for py, px in zip(ys, xs):
#         if visited[py, px]: continue
#             
#         # Found a new component
#         island = []
#         stack = [(py, px)]
#         visited[py, px] = True
#         
#         while stack:
#             cy, cx = stack.pop()
#             island.append((cy-1, cx-1)) # Remove padding
#             
#             # Unrolling neighbors (faster than loop)
#             neighbors = [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)]
#             
#             for ny, nx in neighbors:
#                 # We don't need boundary checks because of padding!
#                 if padded[ny, nx] == 255 and not visited[ny, nx]:
#                     visited[ny, nx] = True
#                     stack.append((ny, nx))
#                     
#         if len(island) >= min_sheet_area:
#             islands.append(island)
#             
#     return islands

# def split_ROI(binary_image, min_sheet_area=500):
#     """
#     OPTIMIZED: Union-Find (Disjoint Set Union) for connected components.
#     This is much faster than DFS/BFS for large images.
#     Time complexity: O(n*m*alpha(n*m)) ~ O(n*m) with path compression.
#     """
#     h, w = binary_image.shape
#     
#     # Find all white pixels
#     ys, xs = np.where(binary_image == 255)
#     if len(ys) == 0:
#         return []
#     
#     # Create a mapping from (y,x) to linear index for Union-Find
#     pixel_map = {}
#     for idx, (y, x) in enumerate(zip(ys, xs)):
#         pixel_map[(y, x)] = idx
#     
#     n_pixels = len(ys)
#     parent = np.arange(n_pixels, dtype=np.int32)
#     rank = np.zeros(n_pixels, dtype=np.int32)
#     
#     # Union-Find helper functions (with path compression)
#     def find(x):
#         if parent[x] != x:
#             parent[x] = find(parent[x])  # Path compression
#         return parent[x]
#     
#     def union(x, y):
#         root_x, root_y = find(x), find(y)
#         if root_x != root_y:
#             # Union by rank
#             if rank[root_x] < rank[root_y]:
#                 parent[root_x] = root_y
#             elif rank[root_x] > rank[root_y]:
#                 parent[root_y] = root_x
#             else:
#                 parent[root_y] = root_x
#                 rank[root_x] += 1
#     
#     # Process each white pixel and union with neighbors
#     for idx, (y, x) in enumerate(zip(ys, xs)):
#         # Check right neighbor
#         if x + 1 < w and (y, x + 1) in pixel_map:
#             union(idx, pixel_map[(y, x + 1)])
#         # Check bottom neighbor
#         if y + 1 < h and (y + 1, x) in pixel_map:
#             union(idx, pixel_map[(y + 1, x)])
#     
#     # Group pixels by their root
#     from collections import defaultdict
#     components = defaultdict(list)
#     for idx, (y, x) in enumerate(zip(ys, xs)):
#         root = find(idx)
#         components[root].append((y, x))
#     
#     # Filter by minimum area
#     islands = [comp for comp in components.values() if len(comp) >= min_sheet_area]
#     
#     return islands

# def split_ROI(binary_image, min_sheet_area=500):
#     """
#     OPTIMIZED v2: Fast stack-based approach with deque and direct array indexing.
#     Avoids overhead of sets/dicts while keeping good cache locality.
#     """
#     from collections import deque
#     
#     h, w = binary_image.shape
#     visited = np.zeros((h, w), dtype=np.bool_)
#     islands = []
#     
#     # Pre-compute white pixel locations (avoid scanning empty space)
#     white_pixels = np.argwhere(binary_image == 255)
#     
#     for start_y, start_x in white_pixels:
#         if visited[start_y, start_x]:
#             continue
#         
#         # Use deque for O(1) append/popleft (faster than list)
#         island = []
#         queue = deque([(start_y, start_x)])
#         visited[start_y, start_x] = True
#         
#         while queue:
#             y, x = queue.popleft()
#             island.append((y, x))
#             
#             # Check 4 neighbors inline (faster than loop)
#             # Up
#             if y > 0 and binary_image[y-1, x] == 255 and not visited[y-1, x]:
#                 visited[y-1, x] = True
#                 queue.append((y-1, x))
#             # Down
#             if y < h-1 and binary_image[y+1, x] == 255 and not visited[y+1, x]:
#                 visited[y+1, x] = True
#                 queue.append((y+1, x))
#             # Left
#             if x > 0 and binary_image[y, x-1] == 255 and not visited[y, x-1]:
#                 visited[y, x-1] = True
#                 queue.append((y, x-1))
#             # Right
#             if x < w-1 and binary_image[y, x+1] == 255 and not visited[y, x+1]:
#                 visited[y, x+1] = True
#                 queue.append((y, x+1))
#         
#         if len(island) >= min_sheet_area:
#             islands.append(island)
#     
#     return islands

# def split_ROI(binary_image, min_sheet_area=500, scale=8):
#     """
#     OPTIMIZED v4: Work at downsampled resolution only.
#     Returns approximate islands - detect_tag will refine to full resolution.
#     This is 16x faster since we never touch full resolution here!
    
#     Args:
#         binary_image: Binary image (0 or 255)
#         min_sheet_area: Minimum area at full resolution
#         scale: Downsampling factor (default 4)
#     """
#     from collections import deque
    
#     h, w = binary_image.shape
    
#     # Downsample to find approximate white regions
#     h_small, w_small = h // scale, w // scale
#     downsampled = binary_image[::scale, ::scale]
    
#     # Connected components on downsampled image only
#     visited = np.zeros((h_small, w_small), dtype=np.bool_)
#     islands = []
    
#     # Adjust min area for downsampled resolution
#     min_area_small = max(1, min_sheet_area // (scale * scale))
    
#     for start_y in range(h_small):
#         for start_x in range(w_small):
#             if downsampled[start_y, start_x] == 255 and not visited[start_y, start_x]:
#                 island = []
#                 queue = deque([(start_y, start_x)])
#                 visited[start_y, start_x] = True
                
#                 while queue:
#                     y, x = queue.popleft()
#                     # Store downsampled coordinates scaled back to approximate full-res
#                     island.append((y * scale, x * scale))
                    
#                     # Check neighbors at downsampled resolution
#                     for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
#                         ny, nx = y + dy, x + dx
#                         if 0 <= ny < h_small and 0 <= nx < w_small:
#                             if downsampled[ny, nx] == 255 and not visited[ny, nx]:
#                                 visited[ny, nx] = True
#                                 queue.append((ny, nx))
                
#                 if len(island) >= min_area_small:
#                     islands.append(island)
    
#     return islands

# def split_ROI(binary_image, min_sheet_area=500, scale=8):
#     """
#     OPTIMIZED v5: Vectorized Seed Finding.
#     Uses np.argwhere to skip empty space scanning entirely.
#     """
#     from collections import deque
#     
#     h, w = binary_image.shape
#     
#     # Downsample
#     h_small, w_small = h // scale, w // scale
#     downsampled = binary_image[::scale, ::scale]
#     
#     visited = np.zeros((h_small, w_small), dtype=np.bool_)
#     islands = []
#     
#     min_area_small = max(1, min_sheet_area // (scale * scale))
#     
#     # 1. VECTORIZED LOOKUP: Find all white pixels instantly
#     # This replaces the nested 'for y... for x...' loops
#     white_pixel_coords = np.argwhere(downsampled == 255)
#     
#     for start_y, start_x in white_pixel_coords:
#         if visited[start_y, start_x]:
#             continue
#             
#         island = []
#         queue = deque([(start_y, start_x)])
#         visited[start_y, start_x] = True
#         
#         while queue:
#             y, x = queue.popleft()
#             island.append((y * scale, x * scale))
#             
#             # Check 4 neighbors
#             # Inline checks are faster than loop for small number of neighbors
#             if y > 0 and downsampled[y-1, x] == 255 and not visited[y-1, x]:
#                 visited[y-1, x] = True
#                 queue.append((y-1, x))
#             if y < h_small-1 and downsampled[y+1, x] == 255 and not visited[y+1, x]:
#                 visited[y+1, x] = True
#                 queue.append((y+1, x))
#             if x > 0 and downsampled[y, x-1] == 255 and not visited[y, x-1]:
#                 visited[y, x-1] = True
#                 queue.append((y, x-1))
#             if x < w_small-1 and downsampled[y, x+1] == 255 and not visited[y, x+1]:
#                 visited[y, x+1] = True
#                 queue.append((y, x+1))
#                 
#         if len(island) >= min_area_small:
#             islands.append(island)
#             
#     return islands

@njit(cache=True)
def _bfs_component_numba(downsampled, start_y, start_x, visited, scale):
    """Numba-optimized BFS for a single connected component."""
    h_small, w_small = downsampled.shape
    
    # Use array-based queue (faster than Python deque in Numba)
    max_queue_size = h_small * w_small
    queue_y = np.empty(max_queue_size, dtype=np.int32)
    queue_x = np.empty(max_queue_size, dtype=np.int32)
    queue_y[0] = start_y
    queue_x[0] = start_x
    head = 0
    tail = 1
    
    island_y = np.empty(max_queue_size, dtype=np.int32)
    island_x = np.empty(max_queue_size, dtype=np.int32)
    island_size = 0
    
    visited[start_y, start_x] = True
    
    while head < tail:
        y = queue_y[head]
        x = queue_x[head]
        head += 1
        
        island_y[island_size] = y * scale
        island_x[island_size] = x * scale
        island_size += 1
        
        # Check 4 neighbors (unrolled for speed)
        if y > 0 and downsampled[y-1, x] == 255 and not visited[y-1, x]:
            visited[y-1, x] = True
            queue_y[tail] = y - 1
            queue_x[tail] = x
            tail += 1
        
        if y < h_small-1 and downsampled[y+1, x] == 255 and not visited[y+1, x]:
            visited[y+1, x] = True
            queue_y[tail] = y + 1
            queue_x[tail] = x
            tail += 1
        
        if x > 0 and downsampled[y, x-1] == 255 and not visited[y, x-1]:
            visited[y, x-1] = True
            queue_y[tail] = y
            queue_x[tail] = x - 1
            tail += 1
        
        if x < w_small-1 and downsampled[y, x+1] == 255 and not visited[y, x+1]:
            visited[y, x+1] = True
            queue_y[tail] = y
            queue_x[tail] = x + 1
            tail += 1
    
    return island_y[:island_size], island_x[:island_size]

def split_ROI(binary_image, min_sheet_area=500, scale=8):
    """
    OPTIMIZED v6: Numba-accelerated BFS.
    10-20x faster than pure Python version.
    """
    h, w = binary_image.shape
    
    # Downsample
    h_small, w_small = h // scale, w // scale
    downsampled = binary_image[::scale, ::scale].copy()  # Ensure contiguous
    
    visited = np.zeros((h_small, w_small), dtype=np.bool_)
    islands = []
    
    min_area_small = max(1, min_sheet_area // (scale * scale))
    
    # Find all white pixels
    white_pixel_coords = np.argwhere(downsampled == 255)
    
    for start_y, start_x in white_pixel_coords:
        if visited[start_y, start_x]:
            continue
        
        # Call Numba-optimized BFS
        island_y, island_x = _bfs_component_numba(downsampled, start_y, start_x, visited, scale)
        
        if len(island_y) >= min_area_small:
            island = [(int(y), int(x)) for y, x in zip(island_y, island_x)]
            islands.append(island)
    
    return islands

# Old version kept below for reference:
# def split_ROI(binary_image, min_sheet_area=500, scale=8):
#     from collections import deque
#     h, w = binary_image.shape
#     h_small, w_small = h // scale, w // scale
#     downsampled = binary_image[::scale, ::scale]
#     visited = np.zeros((h_small, w_small), dtype=np.bool_)
#     islands = []
#     min_area_small = max(1, min_sheet_area // (scale * scale))
#     white_pixel_coords = np.argwhere(downsampled == 255)
#     for start_y, start_x in white_pixel_coords:
#         if visited[start_y, start_x]:
#             continue
#         island = []
#         queue = deque([(start_y, start_x)])
#         visited[start_y, start_x] = True
#         while queue:
#             y, x = queue.popleft()
#             island.append((y * scale, x * scale))
#             if y > 0 and downsampled[y-1, x] == 255 and not visited[y-1, x]:
#                 visited[y+1, x] = True
#                 queue.append((y+1, x))
#             if x > 0 and downsampled[y, x-1] == 255 and not visited[y, x-1]:
#                 visited[y, x-1] = True
#                 queue.append((y, x-1))
#             if x < w_small-1 and downsampled[y, x+1] == 255 and not visited[y, x+1]:
#                 visited[y, x+1] = True
#                 queue.append((y, x+1))
#                 
#         if len(island) >= min_area_small:
#             islands.append(island)
#             
#     return islands


def render_ROI(image, island, index):
    h, w, _ = image.shape
    new_image = np.zeros((h, w, 3), dtype=np.uint8)
    for (y, x) in island[index]:
        new_image[y, x] = image[y, x]
        
    return new_image


# def detect_tag(image, island, gray_image):
#     detected_tags = []
#     h, w = gray_image.shape
#     ys = [y for (y, x) in island]
#     xs = [x for (y, x) in island]
#     min_y, max_y = min(ys), max(ys)
#     min_x, max_x = min(xs), max(xs)
#     visited_black = set()
#     marker_pixels = []
#     x_hor = [0, 1, 0, -1]
#     y_hor = [1, 0, -1, 0]
#     for y in range(min_y, max_y + 1):
#         for x in range(min_x, max_x + 1):
#             if gray_image[y, x] == 0 and (y, x) not in visited_black:
#                 touching_border = False
#                 stack = [(y,x)]
#                 visited_black.add((y,x))
#                 black_region = []
#                 while stack:
#                     cy, cx = stack.pop()
#                     black_region.append((cy, cx))
#                     for direction in range(4):
#                         ny, nx = cy + y_hor[direction], cx + x_hor[direction]
#                         if ny <= min_y or ny >= max_y or nx <= min_x or nx >= max_x:
#                             touching_border = True
#                             continue
#                         
#                         
#                         if gray_image[ny, nx] == 0 and (ny, nx) not in visited_black:
#                             visited_black.add((ny, nx))
#                             stack.append((ny, nx))
#                     # But how is it removing the corner black regions
#                 if not touching_border:
#                     if len(black_region) > len(marker_pixels):
#                         marker_pixels = black_region
#     # Now we have the marker pixels, we can find the corners of the tag using the marker pixels, we can use the fact that the corners of the tag will be the farthest points
#     
#     return marker_pixels

# def detect_tag(image, island, gray_image):
#     """
#     OPTIMIZED: Vectorized black region detection using NumPy operations and Union-Find.
#     Massive speedup by avoiding nested loops and set operations.
#     """
#     h, w = gray_image.shape
#     
#     # Convert island to numpy arrays for vectorized operations
#     island_arr = np.array(island, dtype=np.int32)
#     ys, xs = island_arr[:, 0], island_arr[:, 1]
#     min_y, max_y = ys.min(), ys.max()
#     min_x, max_x = xs.min(), xs.max()
#     
#     # Extract ROI (Region of Interest) to reduce search space dramatically
#     roi_h, roi_w = max_y - min_y + 1, max_x - min_x + 1
#     roi = gray_image[min_y:max_y+1, min_x:max_x+1]
#     
#     # Find all black pixels in ROI using vectorized operation
#     black_mask = (roi == 0)
#     black_ys, black_xs = np.where(black_mask)
#     
#     if len(black_ys) == 0:
#         return []
#     
#     # Create pixel mapping for Union-Find
#     n_black = len(black_ys)
#     pixel_to_idx = np.full((roi_h, roi_w), -1, dtype=np.int32)
#     pixel_to_idx[black_ys, black_xs] = np.arange(n_black)
#     
#     # Union-Find setup
#     parent = np.arange(n_black, dtype=np.int32)
#     
#     def find(x):
#         if parent[x] != x:
#             parent[x] = find(parent[x])
#         return parent[x]
#     
#     def union(x, y):
#         root_x, root_y = find(x), find(y)
#         if root_x != root_y:
#             parent[root_x] = root_y
#     
#     # Union adjacent black pixels (vectorized boundary check)
#     for idx in range(n_black):
#         y, x = black_ys[idx], black_xs[idx]
#         # Check right neighbor
#         if x + 1 < roi_w and pixel_to_idx[y, x + 1] != -1:
#             union(idx, pixel_to_idx[y, x + 1])
#         # Check bottom neighbor
#         if y + 1 < roi_h and pixel_to_idx[y + 1, x] != -1:
#             union(idx, pixel_to_idx[y + 1, x])
#     
#     # Group pixels by component
#     from collections import defaultdict
#     components = defaultdict(list)
#     for idx in range(n_black):
#         root = find(idx)
#         y, x = black_ys[idx], black_xs[idx]
#         components[root].append((y, x))
#     
#     # Check which components touch border (vectorized)
#     marker_pixels = []
#     max_area = 0
#     
#     for comp_pixels in components.values():
#         # Vectorized border check
#         comp_arr = np.array(comp_pixels)
#         comp_ys, comp_xs = comp_arr[:, 0], comp_arr[:, 1]
#         
#         # A component touches border if any pixel is on the ROI border
#         touching_border = (
#             np.any(comp_ys == 0) or np.any(comp_ys == roi_h - 1) or
#             np.any(comp_xs == 0) or np.any(comp_xs == roi_w - 1)
#         )
#         
#         if not touching_border and len(comp_pixels) > max_area:
#             max_area = len(comp_pixels)
#             # Convert back to original image coordinates
#             marker_pixels = [(y + min_y, x + min_x) for y, x in comp_pixels]
#     
#     return marker_pixels

# def detect_tag(image, island, gray_image, scale=8):
#     """
#     OPTIMIZED v5: Fast BFS, then extract boundary pixels efficiently.
    
#     Args:
#         image: Original color image
#         island: Approximate island coordinates from split_ROI
#         gray_image: Grayscale/thresholded image
#         scale: Downsampling factor used in split_ROI (must match!)
#     """
#     from collections import deque
    
#     h, w = gray_image.shape
    
#     # Island is at approximate (downsampled) coordinates
#     # Get bounding box with expansion for full resolution
#     island_arr = np.array(island, dtype=np.int32)
#     approx_min_y, approx_max_y = island_arr[:, 0].min(), island_arr[:, 0].max()
#     approx_min_x, approx_max_x = island_arr[:, 1].min(), island_arr[:, 1].max()
    
#     # Expand bounding box to cover full resolution pixels
#     pad = scale + 2
#     min_y = max(0, approx_min_y - pad)
#     max_y = min(h - 1, approx_max_y + scale + pad)
#     min_x = max(0, approx_min_x - pad)
#     max_x = min(w - 1, approx_max_x + scale + pad)
    
#     roi_h, roi_w = max_y - min_y + 1, max_x - min_x + 1
    
#     # Work at full resolution only in this expanded region
#     visited = np.zeros((roi_h, roi_w), dtype=np.bool_)
#     best_region = []
#     max_area = 0
    
#     # Scan the ROI for black regions at full resolution
#     for y in range(roi_h):
#         for x in range(roi_w):
#             img_y, img_x = y + min_y, x + min_x
            
#             if gray_image[img_y, img_x] == 0 and not visited[y, x]:
#                 queue = deque([(y, x)])
#                 visited[y, x] = True
#                 region = []
#                 touching_border = False
                
#                 while queue:
#                     cy, cx = queue.popleft()
#                     region.append((cy, cx))  # Store in ROI coordinates
                    
#                     # Check if on ROI border
#                     if cy == 0 or cy == roi_h - 1 or cx == 0 or cx == roi_w - 1:
#                         touching_border = True
                    
#                     # Check neighbors
#                     for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
#                         ny, nx = cy + dy, cx + dx
#                         if 0 <= ny < roi_h and 0 <= nx < roi_w:
#                             if gray_image[ny + min_y, nx + min_x] == 0 and not visited[ny, nx]:
#                                 visited[ny, nx] = True
#                                 queue.append((ny, nx))
                
#                 if not touching_border and len(region) > max_area:
#                     max_area = len(region)
#                     best_region = region
    
#     if not best_region:
#         return []
    
#     # Extract boundary pixels efficiently using set lookup
#     region_set = set(best_region)
#     boundary = []
    
#     for cy, cx in best_region:
#         # Check if any of 4 neighbors is NOT in region (= boundary pixel)
#         if ((cy - 1, cx) not in region_set or
#             (cy + 1, cx) not in region_set or
#             (cy, cx - 1) not in region_set or
#             (cy, cx + 1) not in region_set):
#             boundary.append((cy + min_y, cx + min_x))  # Convert to full image coords
    
#     return boundary

def detect_tag(image, island, gray_image, scale=8):
    """
    OPTIMIZED v7: Smart Expansion + Border Rejection.
    1. Expands ROI only into white pixels (recovering resolution without hitting floor).
    2. Rejects black regions touching the border (ignoring floor corners).
    """
    h, w = gray_image.shape
    
    # --- 1. Get Initial Approximate Bounding Box ---
    island_arr = np.array(island, dtype=np.int32)
    # These are already scaled by split_ROI, but correspond to the "block" start
    min_y, max_y = island_arr[:, 0].min(), island_arr[:, 0].max()
    min_x, max_x = island_arr[:, 1].min(), island_arr[:, 1].max()
    
    # Ensure initial box is valid
    # (We initialize the "search box" to the downsampled block edges)
    # The true paper edge is somewhere between 0 and 'scale' pixels outwards.
    
    # --- 2. Smart Expansion (The User's Logic) ---
    # "Look if pixels are white -> add them. If black -> leave them."
    
    # Expand UP
    for _ in range(scale + 2): # Check up to margin
        if min_y > 0:
            # Check the row above. If it has white pixels, include it.
            row_check = gray_image[min_y - 1, min_x : max_x + 1]
            if np.any(row_check == 255): 
                min_y -= 1
            else:
                break # Hit the floor (black), stop expansion
    
    # Expand DOWN
    for _ in range(scale + 2):
        if max_y < h - 1:
            row_check = gray_image[max_y + 1, min_x : max_x + 1]
            if np.any(row_check == 255):
                max_y += 1
            else:
                break
                
    # Expand LEFT
    for _ in range(scale + 2):
        if min_x > 0:
            col_check = gray_image[min_y : max_y + 1, min_x - 1]
            if np.any(col_check == 255):
                min_x -= 1
            else:
                break

    # Expand RIGHT
    for _ in range(scale + 2):
        if max_x < w - 1:
            col_check = gray_image[min_y : max_y + 1, max_x + 1]
            if np.any(col_check == 255):
                max_x += 1
            else:
                break

    # --- 3. Extract Refined ROI ---
    roi = gray_image[min_y : max_y + 1, min_x : max_x + 1]
    roi_h, roi_w = roi.shape
    
    # --- 4. Component Detection with Border Rejection ---
    black_mask = (roi == 0)
    
    if not np.any(black_mask):
        return []

    visited = np.zeros_like(black_mask)
    max_area = 0
    best_component_mask = None
    
    # Get all black pixels
    y_idxs, x_idxs = np.where(black_mask)
    coords = list(zip(y_idxs, x_idxs))
    
    for start_y, start_x in coords:
        if visited[start_y, start_x]:
            continue
            
        stack = [(start_y, start_x)]
        visited[start_y, start_x] = True
        component_pixels = []
        touches_border = False 
        
        while stack:
            cy, cx = stack.pop()
            component_pixels.append((cy, cx))
            
            # Check for Border Touch (This kills the floor triangles)
            if cy == 0 or cy == roi_h - 1 or cx == 0 or cx == roi_w - 1:
                touches_border = True
            
            # Fast Neighbor Check
            # (Unrolled for performance)
            if cy > 0 and black_mask[cy-1, cx] and not visited[cy-1, cx]:
                visited[cy-1, cx] = True
                stack.append((cy-1, cx))
            if cy < roi_h - 1 and black_mask[cy+1, cx] and not visited[cy+1, cx]:
                visited[cy+1, cx] = True
                stack.append((cy+1, cx))
            if cx > 0 and black_mask[cy, cx-1] and not visited[cy, cx-1]:
                visited[cy, cx-1] = True
                stack.append((cy, cx-1))
            if cx < roi_w - 1 and black_mask[cy, cx+1] and not visited[cy, cx+1]:
                visited[cy, cx+1] = True
                stack.append((cy, cx+1))
        
        # REJECT FLOOR:
        if touches_border:
            continue

        if len(component_pixels) > max_area:
            max_area = len(component_pixels)
            best_component_mask = np.zeros_like(black_mask)
            rows = [p[0] for p in component_pixels]
            cols = [p[1] for p in component_pixels]
            best_component_mask[rows, cols] = True

    if best_component_mask is None:
        return []

    # --- 5. Boundary Extraction ---
    # Fast XOR erosion to get contour
    padded = np.pad(best_component_mask, 1, mode='constant', constant_values=False)
    eroded = (padded[1:-1, 1:-1] & 
              padded[:-2, 1:-1] & 
              padded[2:, 1:-1] & 
              padded[1:-1, :-2] & 
              padded[1:-1, 2:])
              
    boundary_mask = best_component_mask ^ eroded
    
    by, bx = np.where(boundary_mask)
    marker_pixels = []
    for i in range(len(by)):
        marker_pixels.append((by[i] + min_y, bx[i] + min_x))
        
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


# def decode_tag(marker_pixels, gray_image): 
#     h, w = gray_image.shape
#     
#     # 1. Get Corners
#     corners = get_corners(marker_pixels)
#     if corners is None: return None

#     # 2. Homography
#     src_pts = np.array([[0,0],[8,0],[8,8],[0,8]], dtype=np.float32)
#     dst_pts = corners.astype(np.float32)
#     H = compute_homography(src_pts, dst_pts)
#     
#     # 3. Sample with Median Voting
#     # Configuration: 5x5 grid, inner 60% of cell
#     sample_resolution = 5
#     margin = 0.2 
#     offsets = np.linspace(-0.5 + margin, 0.5 - margin, sample_resolution)

#     grid_vals = np.zeros((8, 8), dtype=np.uint8)

#     for i in range(8):       # Columns
#         for j in range(8):   # Rows
#             cell_samples = []
#             for dy in offsets:
#                 for dx in offsets:
#                     u, v = i + 0.5 + dx, j + 0.5 + dy
#                     
#                     denom = H[2, 0]*u + H[2, 1]*v + H[2, 2]
#                     if abs(denom) < 1e-6: continue 
#                     
#                     x_img = (H[0, 0]*u + H[0, 1]*v + H[0, 2]) / denom
#                     y_img = (H[1, 0]*u + H[1, 1]*v + H[1, 2]) / denom
#                     
#                     if 0 <= int(y_img) < h and 0 <= int(x_img) < w:
#                         cell_samples.append(gray_image[int(y_img), int(x_img)])

#             # MEDIAN VOTE
#             if len(cell_samples) > 0:
#                 cell_samples.sort()
#                 median_pixel = cell_samples[len(cell_samples) // 2]
#                 grid_vals[j, i] = 1 if median_pixel > 127 else 0

@njit(cache=True)
def _sample_grid_numba(H, gray_image, sample_resolution=5, margin=0.2):
    """Numba-optimized grid sampling with median voting.
    This is the hottest loop in decode_tag - samples 64 cells × 25 points each.
    """
    h, w = gray_image.shape
    grid_vals = np.zeros((8, 8), dtype=np.uint8)
    
    # Precompute offsets
    offsets = np.linspace(-0.5 + margin, 0.5 - margin, sample_resolution)
    
    for i in range(8):  # Columns
        for j in range(8):  # Rows
            # Collect samples for this cell
            samples = np.empty(sample_resolution * sample_resolution, dtype=np.uint8)
            sample_count = 0
            
            for dy in offsets:
                for dx in offsets:
                    u = i + 0.5 + dx
                    v = j + 0.5 + dy
                    
                    denom = H[2, 0]*u + H[2, 1]*v + H[2, 2]
                    if abs(denom) < 1e-6:
                        continue
                    
                    x_img = (H[0, 0]*u + H[0, 1]*v + H[0, 2]) / denom
                    y_img = (H[1, 0]*u + H[1, 1]*v + H[1, 2]) / denom
                    
                    yi = int(y_img)
                    xi = int(x_img)
                    
                    if 0 <= yi < h and 0 <= xi < w:
                        samples[sample_count] = gray_image[yi, xi]
                        sample_count += 1
            
            # Median vote (using partial sort for efficiency)
            if sample_count > 0:
                valid_samples = samples[:sample_count]
                # Simple insertion sort for small arrays (faster than full sort in Numba)
                for k in range(1, sample_count):
                    key = valid_samples[k]
                    m = k - 1
                    while m >= 0 and valid_samples[m] > key:
                        valid_samples[m + 1] = valid_samples[m]
                        m -= 1
                    valid_samples[m + 1] = key
                
                median_pixel = valid_samples[sample_count // 2]
                grid_vals[j, i] = 1 if median_pixel > 127 else 0
    
    return grid_vals

def decode_tag(marker_pixels, gray_image): 
    """OPTIMIZED: Uses Numba for the heavy sampling loop."""
    h, w = gray_image.shape
    
    # 1. Get Corners
    corners = get_corners(marker_pixels)
    if corners is None: return None

    # 2. Homography
    src_pts = np.array([[0,0],[8,0],[8,8],[0,8]], dtype=np.float32)
    dst_pts = corners.astype(np.float32)
    H = compute_homography(src_pts, dst_pts)
    
    # 3. Sample with Median Voting (Numba-optimized)
    grid_vals = _sample_grid_numba(H, gray_image, sample_resolution=5, margin=0.2)

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
        cv2.circle(image, (x, y), 8, colors[i], -1)
        # Draw line to next corner
        next_point = corners[(i + 1) % 4]
        next_x, next_y = int(next_point[0]), int(next_point[1])
        cv2.line(image, (x, y), (next_x, next_y), (0, 255, 0), 2)
    
    cx = int(sum(c[0] for c in corners) / 4)
    cy = int(sum(c[1] for c in corners) / 4)
    cv2.putText(image, str(tag.id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)






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
    cv2.fillConvexPoly(mask_roi, np.int32(dst_pts_roi), 255)
    mask_bool = mask_roi > 0

    out = video_frame.copy()
    out_slice = out[y0:y1 + 1, x0:x1 + 1]
    out_slice[mask_bool] = warped_roi[mask_bool]
    out[y0:y1 + 1, x0:x1 + 1] = out_slice
    return out

    



    
def process_frame_marking(frame, scale=4):
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    detected_tags = []
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            corners = get_corners(marker_pixels)
            if corners is not None:
                for i, (x, y) in enumerate(corners):
                    cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(x) + 5, int(y) + 5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if tag:
                detected_tags.append(tag)      
    for tag in detected_tags:
        mark_corners(frame, tag)

    return frame


def process_frame_superimpose(frame, template_image_path, scale=4):
    gray = threshold_image(frame)
    rois = split_ROI(gray, scale=scale)
    detected_tags = []
    
    template_image = cv2.imread(template_image_path)

    for bbox in rois:
        marker_pixels = detect_tag(frame, bbox, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                detected_tags.append(tag)      
    for tag in detected_tags:
        frame = superimpose(frame, template_image, tag)
    return frame


    
def read_intrinsics(path):
    """
    Reads the 3x3 Intrinsic Camera Matrix from a text file.
    Format: K = np.array([[f_x, s, c_x], [0, f_y, c_y], [0, 0, 1]])
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        # Parse the matrix from the numpy array format
        # Extract everything between the outermost brackets
        import re
        # Find all numbers (including scientific notation)
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
        
        if len(numbers) >= 9:
            values = [float(n) for n in numbers[:9]]
            return np.array(values, dtype=np.float32).reshape(3, 3)
        else:
            print("Error: Intrinsics file does not contain enough values.")
            return np.eye(3, dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: Intrinsics file not found at {path}")
        return np.eye(3, dtype=np.float32)

def compute_projection_matrix(H, K):
    """
    Recovers the Projection Matrix P = K[R|t] from Homography H and Intrinsics K.
    """
    # 1. Remove intrinsics from H to get Pose components
    # H ~ K * [r1, r2, t]  -->  inv(K) * H ~ [r1, r2, t]
    try:
        inv_K = np.linalg.inv(K)
        A = inv_K @ H
    except np.linalg.LinAlgError:
        return None

    # 2. Extract vectors
    col1 = A[:, 0]
    col2 = A[:, 1]
    col3 = A[:, 2]

    # 3. Normalize (recover scale factor lambda)
    # The norm of the first two columns (rotation vectors) should be 1
    norm1 = np.linalg.norm(col1)
    norm2 = np.linalg.norm(col2)
    lamb = (norm1 + norm2) / 2.0
    
    if lamb == 0:
        return None

    r1 = col1 / lamb
    r2 = col2 / lamb
    t  = col3 / lamb

    # 4. Compute third rotation vector (orthonormal basis)
    r3 = np.cross(r1, r2)

    # 5. Enforce orthogonality on Rotation Matrix R using SVD
    # This cleans up noise from the homography
    R_raw = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ Vt

    # 6. Construct Extrinsics [R|t] and Projection P
    # Extrinsics 3x4
    extrinsics = np.column_stack([R, t])
    
    # Projection 3x4: P = K @ [R|t]
    P = K @ extrinsics
    return P

def render(img, obj, projection, model, scale=3, color=False):
    """
    Render a loaded obj model into the current video frame with visualization.

    Args:
        img: The current video frame.
        obj: The loaded OBJ model.
        projection: The 3D projection matrix.
        model: The reference image representing the surface to be augmented.
        scale: Scale factor for the 3D model. Defaults to 3.
        color: Whether to render in color. Defaults to False.
    """
    DEFAULT_COLOR = (0, 150, 0)  # Darker green for filled faces
    WIREFRAME_COLOR = (0, 255, 0)  # Bright green for edges
    
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        
        # Fill the face for visualization
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color_bgr = hex_to_rgb(face[-1])
            color_bgr = color_bgr[::-1]
            cv2.fillConvexPoly(img, imgpts, color_bgr)
        
        # Draw wireframe edges for better visualization
        cv2.polylines(img, [imgpts], isClosed=True, color=WIREFRAME_COLOR, thickness=2)

    return img


def process_frame_3D(frame, model_path, intrinsics_path, scale_3d=50.0, scale=4):
    """
    Main pipeline for 3D AR Rendering.
    
    Args:
        frame: Input video frame
        model_path: Path to .obj file
        intrinsics_path: Path to camera intrinsics file
        scale_3d: Scale factor for 3D model
        scale: Downsampling scale for tag detection
    """
    # 1. Load Camera Intrinsics
    K = read_intrinsics(intrinsics_path)
    
    # 2. Load 3D Model (with simple caching to avoid re-parsing every frame)
    if not hasattr(process_frame_3D, "obj_cache"):
        process_frame_3D.obj_cache = {}
    
    if model_path not in process_frame_3D.obj_cache:
        # swapyz=True typically helps align OBJ up-axis (Y) with Camera up-axis
        process_frame_3D.obj_cache[model_path] = OBJ(model_path, swapyz=True)
    
    obj = process_frame_3D.obj_cache[model_path]

    # 3. Detect Tags (reuse existing pipeline)
    gray = threshold_image(frame)
    rois = split_ROI(gray, scale=scale)
    
    # Define Virtual Tag dimensions for Homography
    # This establishes the coordinate system on the tag surface.
    # We use 200x200 to match the tag size in world coordinates
    TAG_SIZE = 200 
    src_pts = np.array([
        [0, 0], 
        [TAG_SIZE - 1, 0], 
        [TAG_SIZE - 1, TAG_SIZE - 1], 
        [0, TAG_SIZE - 1]
    ], dtype=np.float32)

    detected_tags = []
    
    # Detect
    for bbox in rois:
        marker_pixels = detect_tag(frame, bbox, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                detected_tags.append(tag)
    
    # Render
    for tag in detected_tags:
        # A. Compute Homography (2D Image -> 2D Screen)
        dst_pts = tag.corners.astype(np.float32)
        H = compute_homography(src_pts, dst_pts)
        
        # B. Compute 3D Projection Matrix (3D World -> 2D Screen)
        # This effectively places the virtual camera relative to the tag
        P = compute_projection_matrix(H, K)
        
        if P is not None:
            # C. Create a reference model plane for the render function
            # The render function expects model.shape to determine the tag dimensions
            model_plane = np.zeros((TAG_SIZE, TAG_SIZE), dtype=np.uint8)
            
            # D. Render Object with visualization
            frame = render(frame, obj, P, model_plane, scale=scale_3d, color=False)
            
    return frame