import cv2
import numpy as np
from numba import njit
from collections import deque
# Import your manual optimized functions (only the ones actually used)
from manual_cv2 import (
    convex_hull_contour,
    approx_poly_dp,
    arc_length,
    # fill_convex_poly,
    convex_hull,
    warp_perspective,
)


class KalmanCornerTracker:
    """
    Kalman filter for tracking a single corner point with constant velocity model.
    State: [x, y, vx, vy] - position and velocity
    """
    def __init__(self, initial_pos, process_noise=1.0, measurement_noise=5.0):
        """
        Args:
            initial_pos: Initial [x, y] position
            process_noise: Process noise (motion model uncertainty)
            measurement_noise: Measurement noise (detection uncertainty)
        """
        # State: [x, y, vx, vy]
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float32)
        
        # State covariance matrix (uncertainty in state estimate)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        
        # State transition matrix (constant velocity model)
        # x_new = x + vx*dt, y_new = y + vy*dt, vx_new = vx, vy_new = vy
        dt = 1.0  # Assuming constant frame rate
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we only measure position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.Q[2:, 2:] *= 2.0  # Higher uncertainty in velocity
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
    
    def predict(self):
        """Predict next state using motion model."""
        # State prediction
        self.state = self.F @ self.state
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement):
        """Update state with new measurement."""
        measurement = np.array(measurement, dtype=np.float32)
        
        # Innovation (measurement residual)
        y = measurement - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.state = self.state + K @ y
        
        # Covariance update
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[:2]  # Return updated position
    
    def get_position(self):
        """Get current position estimate."""
        return self.state[:2].copy()
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.state[2:].copy()


class KalmanTagTracker:
    """Tracks AR tag corners using Kalman filters with constant velocity motion model."""
    def __init__(self, process_noise=2.0, measurement_noise=5.0, max_missing_frames=5, max_innovation=150, orientation_alpha=0.3):
        """
        Args:
            process_noise: Process noise for Kalman filter (motion uncertainty).
            measurement_noise: Measurement noise (detection uncertainty).
            max_missing_frames: Maximum frames before removing tag from tracking.
            max_innovation: Maximum innovation (measurement - prediction) to accept measurement.
            orientation_alpha: Smoothing factor for orientation (0=no update, 1=no smoothing).
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_missing_frames = max_missing_frames
        self.max_innovation = max_innovation
        self.orientation_alpha = orientation_alpha
        self.tag_trackers = {}  # {tag_id: {'filters': [4 KalmanCornerTrackers], 'frames_missing': int, 'orientation': float}}
    
    def track_corners(self, tag_id, measured_corners):
        """
        Track tag corners with Kalman filtering.
        
        Args:
            tag_id: Integer ID of the detected tag.
            measured_corners: Newly detected corners (4x2 array).
        
        Returns:
            Smoothed corners (4x2 array).
        """
        if tag_id not in self.tag_trackers:
            # Initialize Kalman filters for each corner
            filters = [
                KalmanCornerTracker(measured_corners[i], self.process_noise, self.measurement_noise)
                for i in range(4)
            ]
            self.tag_trackers[tag_id] = {
                'filters': filters,
                'frames_missing': 0,
                'orientation': None  # Will be set when orientation is provided
            }
            return measured_corners  # Return initial measurement as-is
        
        tag_data = self.tag_trackers[tag_id]
        filters = tag_data['filters']
        smoothed_corners = np.zeros((4, 2), dtype=np.float32)
        
        for i in range(4):
            # Predict next position
            predicted_pos = filters[i].predict()
            
            # Check innovation (difference between measurement and prediction)
            innovation = np.linalg.norm(measured_corners[i] - predicted_pos)
            
            if innovation > self.max_innovation:
                # Measurement too far from prediction - likely wrong association or occlusion
                # Reinitialize this corner's filter
                filters[i] = KalmanCornerTracker(
                    measured_corners[i], 
                    self.process_noise, 
                    self.measurement_noise
                )
                smoothed_corners[i] = measured_corners[i]
            else:
                # Update with measurement
                smoothed_corners[i] = filters[i].update(measured_corners[i])
        
        tag_data['frames_missing'] = 0

        # ---- RIGID QUAD ENFORCEMENT ----
        c = np.mean(smoothed_corners, axis=0)

        # Edge vectors from corner 0
        v1 = smoothed_corners[1] - smoothed_corners[0]
        v2 = smoothed_corners[3] - smoothed_corners[0]

        # Reconstruct corner 2 rigidly
        smoothed_corners[2] = smoothed_corners[0] + v1 + v2

        return smoothed_corners
    
    def smooth_orientation(self, tag_id, measured_orientation):
        """
        Smooth orientation angle using exponential moving average.
        Handles wrap-around at 0/360 degrees properly.
        
        Args:
            tag_id: Integer ID of the tag
            measured_orientation: New orientation measurement in degrees (0-360)
        
        Returns:
            Smoothed orientation in degrees (0-360)
        """
        if tag_id not in self.tag_trackers:
            return measured_orientation
        
        tag_data = self.tag_trackers[tag_id]
        
        # Initialize if first time
        if tag_data['orientation'] is None:
            tag_data['orientation'] = measured_orientation
            return measured_orientation
        
        prev_orientation = tag_data['orientation']
        
        # Handle wrap-around: convert to circular mean
        # Use vectors to avoid 0/360 discontinuity
        prev_rad = np.deg2rad(prev_orientation)
        meas_rad = np.deg2rad(measured_orientation)
        
        # Exponential moving average in vector space
        prev_x = np.cos(prev_rad)
        prev_y = np.sin(prev_rad)
        meas_x = np.cos(meas_rad)
        meas_y = np.sin(meas_rad)
        
        # Smooth
        smooth_x = (1 - self.orientation_alpha) * prev_x + self.orientation_alpha * meas_x
        smooth_y = (1 - self.orientation_alpha) * prev_y + self.orientation_alpha * meas_y
        
        # Convert back to angle
        smooth_orientation = np.rad2deg(np.arctan2(smooth_y, smooth_x))
        
        # Normalize to 0-360
        if smooth_orientation < 0:
            smooth_orientation += 360.0
        
        tag_data['orientation'] = smooth_orientation
        return smooth_orientation
   
    def predict_missing_tag(self, tag_id):
        """
        Predict tag position when not detected (for brief occlusions).
        
        Args:
            tag_id: Integer ID of the tag.
        
        Returns:
            Predicted corners (4x2 array) or None if tag not tracked.
        """
        if tag_id not in self.tag_trackers:
            return None
        
        filters = self.tag_trackers[tag_id]['filters']
        predicted_corners = np.zeros((4, 2), dtype=np.float32)
        
        for i in range(4):
            predicted_corners[i] = filters[i].predict()
        
        return predicted_corners
    
    def update_missing_tags(self, detected_tag_ids):
        """
        Update tracking for tags that weren't detected in current frame.
        
        Args:
            detected_tag_ids: Set of tag IDs detected in current frame.
        """
        ids_to_remove = []
        for tag_id in self.tag_trackers:
            if tag_id not in detected_tag_ids:
                self.tag_trackers[tag_id]['frames_missing'] += 1
                if self.tag_trackers[tag_id]['frames_missing'] > self.max_missing_frames:
                    ids_to_remove.append(tag_id)
        
        for tag_id in ids_to_remove:
            del self.tag_trackers[tag_id]
    
    def reset(self):
        """Clear all tracking history."""
        self.tag_trackers.clear()


# Global Kalman trackers for different processing modes - Tuned for smoother tracking
_tracker_superimpose = KalmanTagTracker(process_noise=0.2, measurement_noise=30.0, max_innovation=150, orientation_alpha=0.05)
_tracker_3d = KalmanTagTracker(process_noise=0.5, measurement_noise=12.0, max_innovation=150, orientation_alpha=0.05)
_tracker_marking = KalmanTagTracker(process_noise=0.8, measurement_noise=10.0, max_innovation=150, orientation_alpha=0.05)
_depth_memory = {}  

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
    """
    Convert frame to binary using Otsu's automatic thresholding.
    Better than adaptive for high-contrast AR tags.
    """
    # Convert to grayscale
    gray = 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
    gray = gray.astype(np.uint8)
    
    # Apply slight Gaussian blur to reduce noise before thresholding
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding - automatically finds optimal threshold
    # Works excellently for bimodal histograms (AR tags: black vs white)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
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
    
    # Robust corner ordering: always start from top-left, go clockwise
    # 1. Find center
    center = np.mean(corners, axis=0)
    
    # 2. Sort by angle (counter-clockwise from positive x-axis)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    corners = corners[sorted_indices]
    
    # 3. Find top-left corner robustly: minimum x+y value
    # This is geometrically stable - top-left always has smallest sum
    sum_coords = corners[:, 0] + corners[:, 1]
    top_left_idx = np.argmin(sum_coords)
    
    # 4. Roll so top-left is first
    corners = np.roll(corners, -top_left_idx, axis=0)
    
    return corners.astype(np.float32)

class ARtag:
    def __init__(self, corners, id, orientation_steps, visual_orientation=None, total_orientation=None):
        self.corners = corners
        self.id = id
        self.orientation_steps = orientation_steps  # Discrete pattern rotation (0-3)
        self.rotation = orientation_steps * 90  # Pattern rotation in degrees
        self.visual_orientation = visual_orientation  # 2D visual rotation from detected corners
        self.total_orientation = total_orientation  # Total = visual + pattern rotation

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

def compute_orientation_from_homography(H, K):
    """
    Compute the visual 2D orientation angle of the tag from homography.
    This is the in-plane rotation angle of the tag as it appears in the image.
    
    Args:
        H: Homography matrix (3x3) from canonical tag coords to detected corners
        K: Camera intrinsics matrix (3x3)
    
    Returns:
        Orientation angle in degrees (0-360), or None if computation fails
    """
    try:
        inv_K = np.linalg.inv(K)
        A = inv_K @ H
        
        # Extract rotation columns
        col1 = A[:, 0]
        col2 = A[:, 1]
        
        # Normalize to get rotation vectors
        norm = (np.linalg.norm(col1) + np.linalg.norm(col2)) / 2.0
        if norm == 0:
            return None
            
        r1 = col1 / norm
        r2 = col2 / norm
        r3 = np.cross(r1, r2)
        
        # Construct rotation matrix
        R = np.column_stack([r1, r2, r3])
        
        # Orthogonalize using SVD
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # Extract orientation angle from rotation matrix
        # This is the Z-axis rotation (yaw) - how the tag is rotated in the image plane
        angle = np.arctan2(R[1, 0], R[0, 0]) * 180.0 / np.pi
        
        # Normalize to 0-360 range
        if angle < 0:
            angle += 360.0
            
        return angle
    except:
        return None

def compute_visual_orientation_simple(corners):
    """
    Compute the simple 2D visual orientation from corner positions.
    This is the angle the marker needs to rotate to align with canonical orientation.
    Uses edge vectors for robustness against corner ordering noise.
    
    Args:
        corners: 4x2 array of corner positions (ordered: top-left, top-right, bottom-right, bottom-left)
    
    Returns:
        Orientation angle in degrees (0-360)
    """
    # Use edge vectors instead of single corner - much more robust!
    # Average the orientation from two edges to reduce noise
    
    # Edge from corner 0 to corner 1 (top edge)
    edge1 = corners[1] - corners[0]
    angle1 = np.arctan2(edge1[1], edge1[0]) * 180.0 / np.pi
    
    # Edge from corner 3 to corner 0 (left edge) - should be perpendicular
    edge2 = corners[0] - corners[3]
    angle2 = np.arctan2(edge2[1], edge2[0]) * 180.0 / np.pi
    
    # Use the top edge angle as primary orientation
    # In canonical orientation, top edge points right (0°)
    visual_angle = angle1
    
    # Normalize to 0-360 range
    while visual_angle < 0:
        visual_angle += 360.0
    while visual_angle >= 360:
        visual_angle -= 360.0
    
    return visual_angle

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
    
    # Don't compute orientation yet - will do it after corner smoothing
    # This ensures orientation is consistent with smoothed corners
    visual_orientation = None
    
    if rotation > 0:
        grid_vals = np.rot90(grid_vals, k=rotation)
        corners = np.roll(corners, shift=-rotation, axis=0)

    found_id = (grid_vals[3,3]*8) + (grid_vals[3,4]*4) + (grid_vals[4,4]*2) + (grid_vals[4,3]*1)
    
    # Total orientation will be computed later from smoothed corners
    total_orientation = None
    
    return ARtag(corners, found_id, rotation, visual_orientation, total_orientation)

def compute_projection_matrix(corners, K, tag_id = None, alpha = 0.1):
    src_pts = np.array([[0,0], [200,0], [200,200], [0,200]], dtype=np.float32)
    H = compute_homography_manual(src_pts, corners)
    try:
        inv_K = np.linalg.inv(K)
        A = inv_K @ H
        col1 = A[:, 0]; col2 = A[:, 1]; col3 = A[:, 2]
        raw_norm = (np.linalg.norm(col1) + np.linalg.norm(col2)) / 2.0
        if raw_norm == 0: return None
        if tag_id is not None:
            if tag_id not in _depth_memory:
                _depth_memory[tag_id] = raw_norm
            else:
                _depth_memory[tag_id] = alpha * raw_norm + (1 - alpha) * _depth_memory[tag_id]
            norm = _depth_memory[tag_id]
                
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
    """
    Read camera intrinsics from file.
    Supports .npz (numpy format) and .txt (plain text) formats.
    """
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
    
    try:
        # Try numpy .npz format first (from calibration script)
        if path.endswith('.npz'):
            data = np.load(path)
            if 'camera_matrix' in data:
                K = data['camera_matrix'].astype(np.float32)
                return K
        
        # Fall back to text format
        with open(path, 'r') as f:
            content = f.read()
            import re
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
            if len(nums) >= 9:
                K = np.array(nums[:9], dtype=np.float32).reshape(3, 3)
    except Exception as e:
        print(f"Warning: Could not read intrinsics from {path}, using default. Error: {e}")
    
    return K


_obj_cache = {}

def process_frame_3D(frame, model_path, intrinsics_path, scale_3d=50.0, scale=4, smoother=None):
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    K = read_intrinsics(intrinsics_path)
    if model_path not in _obj_cache:
        _obj_cache[model_path] = OBJ(model_path, swapyz=True)
    obj = _obj_cache[model_path]
    
    # Use provided smoother or default global tracker
    tracker = smoother if smoother is not None else _tracker_3d
    detected_tag_ids = set()
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                detected_tag_ids.add(tag.id)
                
                # Apply Kalman tracking
                tracked_corners = tracker.track_corners(tag.id, tag.corners)
                
                P = compute_projection_matrix(tracked_corners, K, tag_id=tag.id, alpha=0.1)
                if P is not None:
                    frame = render(frame, obj, P, scale=scale_3d)
    
    # Update missing tags
    tracker.update_missing_tags(detected_tag_ids)
    
    return frame

def process_frame_marking(frame, scale=4, smoother=None):
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    detected_tags = []
    detected_tag_ids = set()
    
    # Use provided smoother or default global tracker
    tracker = smoother if smoother is not None else _tracker_marking
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                detected_tag_ids.add(tag.id)
                
                # Apply Kalman tracking to corners FIRST
                tag.corners = tracker.track_corners(tag.id, tag.corners)
                
                # Compute orientation from NORMALIZED, SMOOTHED corners
                # These corners are already in canonical order (pattern rotation applied)
                # So the visual orientation directly tells us the total orientation
                tag.visual_orientation = compute_visual_orientation_simple(tag.corners)
                tag.total_orientation = tag.visual_orientation  # Already accounts for pattern rotation
                
                # Smooth orientation angle
                if tag.total_orientation is not None:
                    tag.total_orientation = tracker.smooth_orientation(tag.id, tag.total_orientation)
                
                detected_tags.append(tag)
    
    # Update missing tags
    tracker.update_missing_tags(detected_tag_ids)
    
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
        
        # Display ID and total orientation
        if tag.total_orientation is not None:
            text = f"ID:{tag.id} {tag.total_orientation:.1f}°"
        else:
            text = f"ID:{tag.id} {tag.rotation}°"
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
    return frame

def process_frame_superimpose(frame, template_path, scale=4, smoother=None):
    template = cv2.imread(template_path)
    if template is None: return frame
    gray = threshold_image(frame)
    islands = split_ROI(gray, scale=scale)
    h_temp, w_temp = template.shape[:2]
    src_pts = np.array([[0,0], [w_temp,0], [w_temp,h_temp], [0,h_temp]], dtype=np.float32)
    
    frame_h, frame_w = frame.shape[:2]
    detected_tag_ids = set()
    
    # Use provided smoother or default global tracker
    tracker = smoother if smoother is not None else _tracker_superimpose
    
    for island in islands:
        marker_pixels = detect_tag(frame, island, gray, scale=scale)
        if marker_pixels:
            tag = decode_tag(marker_pixels, gray)
            if tag:
                detected_tag_ids.add(tag.id)
                
                # Apply Kalman tracking
                tracked_corners = tracker.track_corners(tag.id, tag.corners)
                
                H = compute_homography_manual(src_pts, tracked_corners)
                warped = warp_perspective(template, H, (frame_w, frame_h))
                mask_src = np.ones((h_temp, w_temp, 1), dtype=np.uint8) * 255
                mask_warped = warp_perspective(mask_src, H, (frame_w, frame_h))
                mask_bool = mask_warped[:, :, 0] > 0
                frame[mask_bool] = warped[mask_bool]
    
    # Update missing tags
    tracker.update_missing_tags(detected_tag_ids)
                
    return frame

