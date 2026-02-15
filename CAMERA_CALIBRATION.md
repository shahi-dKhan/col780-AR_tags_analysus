# Camera Calibration Guide

This guide explains how to calibrate your webcam for accurate 3D AR overlay.

## Quick Start

### Option 1: Full Automatic (Capture + Calibrate)
```bash
cd src
python calibrate_camera.py --mode both
```
This will:
1. Open your webcam
2. Let you capture 20 calibration images
3. Automatically calibrate
4. Save intrinsics to `assets/videos_and_intrinsics/webcam_intrinsics.npz`

### Option 2: Use Existing Calibration Pattern (from calib.io)
If you already have calibration images in `assets`:
```bash
python calibrate_camera.py --mode calibrate --images "assets/calib_*.jpg"
```

## Step-by-Step Instructions

### 1. Prepare Calibration Pattern

**Print a chessboard pattern:**
- Visit https://calib.io/pages/camera-calibration-pattern-generator
- Or use OpenCV's built-in patterns
- Default: 9x6 internal corners (10x7 squares)
- Print on A4/Letter paper
- Mount on flat rigid surface (cardboard works well)

### 2. Capture Calibration Images

Run the capture script:
```bash
python calibrate_camera.py --mode capture --num-images 20
```

**Tips for good calibration:**
- Hold the chessboard pattern in front of webcam
- Vary the position: left, right, center, top, bottom
- Vary the angle: tilted, rotated, different orientations
- Vary the distance: near and far (fill 30-70% of frame)
- Keep the pattern flat and well-lit
- Wait for pattern detection (green corners) before pressing SPACE
- Capture at least 15-20 good images

**During capture:**
- Pattern detected → corners turn **GREEN** → Press **SPACE** to capture
- Pattern not detected → text turns **RED** → adjust position/lighting
- Press **'q'** to quit early

### 3. Calibrate Camera

If you captured images separately:
```bash
python calibrate_camera.py --mode calibrate
```

The script will:
- Load all images from `calib_images/`
- Detect chessboard corners
- Compute camera matrix and distortion coefficients
- Calculate reprojection error
- Save results to `assets/videos_and_intrinsics/webcam_intrinsics.npz`

### 4. Verify Calibration Quality

**Good calibration:**
- RMS Reprojection Error < 0.5 pixels (excellent)
- RMS Reprojection Error < 1.0 pixels (good)
- Mean Error < 0.3 pixels

**Bad calibration (recalibrate if):**
- RMS Error > 1.5 pixels
- Very few successful images (< 10)
- Corners weren't detected accurately

### 5. Use Calibrated Camera

Update your run command to use the new intrinsics:
```bash
python src/main.py --intrinsics assets/videos_and_intrinsics/webcam_intrinsics.npz --model path/to/model.obj
```

The `read_intrinsics()` function now automatically supports `.npz` format!

## Advanced Options

### Custom Chessboard Size
If your pattern has different dimensions:
```bash
python calibrate_camera.py --checkerboard 7 5 --mode both
```
(for 7 columns × 5 rows of **internal corners**)

### Specify Square Size
If you know the actual square size (e.g., 25mm):
```bash
python calibrate_camera.py --square-size 25.0 --mode both
```
This gives real-world scale in calibration results.

### Custom Output Location
```bash
python calibrate_camera.py --output-file my_camera_calib.npz
```

### Use Different Camera
Edit line in `calibrate_camera.py`:
```python
cap = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc.
```

## Troubleshooting

**Pattern not detected:**
- Ensure good lighting (no shadows, no glare)
- Keep pattern completely flat
- Pattern must fill 20-80% of frame
- All corners must be visible
- Check the checkerboard size parameter matches your printed pattern

**High reprojection error:**
- Recapture with more varied positions/angles
- Ensure pattern stays flat during capture
- Use more images (30-40)
- Verify correct checkerboard size parameter

**Webcam not opening:**
- Check if another app is using the camera
- Try different camera index: `--camera-id 1`
- Check permissions (especially on macOS)

## File Formats

**Output `.npz` file contains:**
- `camera_matrix`: 3x3 camera intrinsic matrix K
- `dist_coeffs`: Distortion coefficients [k1, k2, p1, p2, k3]

**Output `.txt` file contains:**
- Human-readable version of calibration results
- Useful for debugging and manual inspection

## Example: Complete Workflow

```bash
# 1. Navigate to src directory
cd src

# 2. Capture 25 images with your printed 9x6 chessboard
python calibrate_camera.py --mode capture --num-images 25

# 3. Calibrate from captured images
python calibrate_camera.py --mode calibrate

# 4. Check the results
cat assets/videos_and_intrinsics/webcam_intrinsics.txt

# 5. Test with your video
python main.py --intrinsics assets/videos_and_intrinsics/webcam_intrinsics.npz \
               --model path/to/model.obj

# 6. Or use webcam live
python main.py --intrinsics assets/videos_and_intrinsics/webcam_intrinsics.npz \
               --model path/to/model.obj
```

## References

- OpenCV Camera Calibration Tutorial: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Calibration Pattern Generator: https://calib.io/
- Zhang's Calibration Method (theory): https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf
