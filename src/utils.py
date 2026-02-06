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
    thresh = 200
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
            