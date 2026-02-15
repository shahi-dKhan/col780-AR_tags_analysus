#!/usr/bin/env python3
"""
Camera Calibration Utility for Webcam
Calibrates camera intrinsics and distortion coefficients using a chessboard pattern.
"""

import numpy as np
import cv2
import glob
import os
import argparse
from pathlib import Path


def calibrate_from_images(image_pattern, checkerboard_size=(9, 6), square_size=1.0, visualize=True):
    """
    Calibrate camera from a set of images containing a chessboard pattern.
    
    Args:
        image_pattern: Glob pattern for calibration images (e.g., 'calib_*.jpg')
        checkerboard_size: Tuple (cols, rows) of internal corners in chessboard
        square_size: Size of a square in your defined unit (mm, cm, etc.)
        visualize: Whether to display detected corners
    
    Returns:
        Camera matrix, distortion coefficients, and reprojection error
    """
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(cols-1,rows-1,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by square size
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    images = glob.glob(image_pattern)
    
    if not images:
        print(f"No images found matching pattern: {image_pattern}")
        return None, None, None
    
    print(f"Found {len(images)} images")
    successful_images = 0
    
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # If found, add object points and image points (after refining them)
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            successful_images += 1
            print(f"✓ Pattern found in: {os.path.basename(fname)}")
            
            # Draw and display the corners
            if visualize:
                cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
                cv2.imshow('Calibration - Press any key to continue', img)
                cv2.waitKey(500)
        else:
            print(f"✗ Pattern NOT found in: {os.path.basename(fname)}")
    
    if visualize:
        cv2.destroyAllWindows()
    
    if successful_images == 0:
        print("No valid calibration images found!")
        return None, None, None
    
    print(f"\nCalibrating camera using {successful_images} images...")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print(f"\n{'='*60}")
    print("Camera Calibration Results:")
    print(f"{'='*60}")
    print(f"RMS Reprojection Error: {ret:.4f} pixels")
    print(f"Mean Reprojection Error: {mean_error:.4f} pixels")
    print(f"\nCamera Matrix (K):")
    print(camera_matrix)
    print(f"\nDistortion Coefficients:")
    print(dist_coeffs.ravel())
    print(f"{'='*60}\n")
    
    return camera_matrix, dist_coeffs, mean_error


def capture_calibration_images(output_dir, checkerboard_size=(9, 6), num_images=20):
    """
    Capture calibration images from webcam.
    
    Args:
        output_dir: Directory to save captured images
        checkerboard_size: Tuple (cols, rows) of internal corners
        num_images: Number of images to capture
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print(f"\n{'='*60}")
    print("Calibration Image Capture Mode")
    print(f"{'='*60}")
    print(f"Target: {num_images} images with detected pattern")
    print(f"Checkerboard size: {checkerboard_size[0]}x{checkerboard_size[1]} internal corners")
    print("\nInstructions:")
    print("  - Move the checkerboard to different positions and angles")
    print("  - Press SPACE when pattern is detected (green) to capture")
    print("  - Press 'q' to quit early")
    print(f"{'='*60}\n")
    
    captured_count = 0
    frame_count = 0
    
    while captured_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # Display frame with status
        display = frame.copy()
        
        if found:
            cv2.drawChessboardCorners(display, checkerboard_size, corners, found)
            status_color = (0, 255, 0)  # Green
            status_text = "Pattern Detected - Press SPACE to capture"
        else:
            status_color = (0, 0, 255)  # Red
            status_text = "Searching for pattern..."
        
        # Draw status
        cv2.putText(display, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(display, f"Captured: {captured_count}/{num_images}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Camera Calibration - Capture Images', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capture image on SPACE if pattern detected
        if key == ord(' ') and found:
            filename = os.path.join(output_dir, f"calib_{captured_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            captured_count += 1
            print(f"Captured image {captured_count}/{num_images}: {filename}")
            cv2.waitKey(200)  # Brief pause to prevent double capture
        
        # Quit on 'q'
        elif key == ord('q'):
            print(f"\nCapture stopped. Captured {captured_count} images.")
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCapture complete! {captured_count} images saved to {output_dir}")
    return captured_count > 0


def save_calibration(camera_matrix, dist_coeffs, output_file):
    """Save calibration data to file."""
    np.savez(output_file, 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs)
    print(f"Calibration data saved to: {output_file}")
    
    # Also save in text format for easy viewing
    txt_file = output_file.replace('.npz', '.txt')
    with open(txt_file, 'w') as f:
        f.write("Camera Matrix (K):\n")
        f.write(str(camera_matrix) + "\n\n")
        f.write("Distortion Coefficients:\n")
        f.write(str(dist_coeffs.ravel()) + "\n")
    print(f"Calibration data (readable) saved to: {txt_file}")


def load_calibration(npz_file):
    """Load calibration data from file."""
    data = np.load(npz_file)
    return data['camera_matrix'], data['dist_coeffs']


def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Utility")
    parser.add_argument('--mode', choices=['capture', 'calibrate', 'both'], default='both',
                       help='Mode: capture images, calibrate from existing, or both')
    parser.add_argument('--images', type=str, default='calib_images/calib_*.jpg',
                       help='Glob pattern for calibration images')
    parser.add_argument('--output-dir', type=str, default='calib_images',
                       help='Directory for captured images')
    parser.add_argument('--output-file', type=str, 
                       default='assets/videos_and_intrinsics/webcam_intrinsics.npz',
                       help='Output file for calibration data')
    parser.add_argument('--checkerboard', type=int, nargs=2, default=[9, 6],
                       help='Checkerboard size (cols rows) - internal corners')
    parser.add_argument('--square-size', type=float, default=1.0,
                       help='Size of checkerboard square in your unit (mm/cm)')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of calibration images to capture')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization during calibration')
    
    args = parser.parse_args()
    
    checkerboard_size = tuple(args.checkerboard)
    
    # Capture mode
    if args.mode in ['capture', 'both']:
        print("\n=== CAPTURE MODE ===")
        success = capture_calibration_images(
            args.output_dir, 
            checkerboard_size, 
            args.num_images
        )
        if not success and args.mode == 'both':
            print("Capture failed, skipping calibration")
            return
    
    # Calibration mode
    if args.mode in ['calibrate', 'both']:
        print("\n=== CALIBRATION MODE ===")
        
        # Use captured images if in 'both' mode
        if args.mode == 'both':
            image_pattern = os.path.join(args.output_dir, 'calib_*.jpg')
        else:
            image_pattern = args.images
        
        camera_matrix, dist_coeffs, error = calibrate_from_images(
            image_pattern,
            checkerboard_size,
            args.square_size,
            visualize=not args.no_visualize
        )
        
        if camera_matrix is not None:
            # Create output directory if needed
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            save_calibration(camera_matrix, dist_coeffs, args.output_file)
            
            print("\n✓ Camera calibration complete!")
            print(f"  Use this file in your AR application: {args.output_file}")
        else:
            print("\n✗ Calibration failed!")


if __name__ == '__main__':
    main()
