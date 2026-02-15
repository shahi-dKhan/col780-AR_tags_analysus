# COL780 - AR Tags Analysis and Augmented Reality

**Author:** Shahid Khan (ee1221163@iitd.ac.in)  
**Course:** COL780 - Computer Vision  
**Repository:** [https://github.com/shahi-dKhan/col780-AR_tags_analysus](https://github.com/shahi-dKhan/col780-AR_tags_analysus)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [Usage Instructions](#usage-instructions)
- [Input/Output Specifications](#inputoutput-specifications)
- [Demo Instructions](#demo-instructions)
- [Implementation Details](#implementation-details)
- [Performance](#performance)
- [Known Limitations](#known-limitations)

---

## Overview

This project implements a complete AR (Augmented Reality) Tag detection and tracking system from scratch, including:

1. **Detection & Identification**: Custom implementation of AR tag detection using flood-fill, convex hull, Douglas-Peucker approximation, and homography-based decoding
2. **2D Augmented Reality**: Real-time image overlay on detected tags with Kalman filtering for smooth tracking
3. **3D Augmented Reality**: 3D object rendering on AR tags using pose estimation via homography decomposition

The implementation uses **custom algorithms** for most operations (no OpenCV ArUco detector), with heavy optimization using NumPy vectorization and Numba JIT compilation to achieve ~12 FPS on 1920×1080 video.

---

## Features

### Core Functionality
- ✅ Custom AR tag detection (no OpenCV ArUco)
- ✅ 4-bit tag ID decoding with orientation detection
- ✅ Multi-tag support in single frame
- ✅ Real-time 2D image overlay with perspective warping
- ✅ 3D model rendering (.obj format) with pose estimation
- ✅ Kalman filtering for temporal smoothing and jitter reduction
- ✅ Webcam support with camera calibration utility
- ✅ Optimized processing (~12 FPS on 1080p video)

### Technical Highlights
- **Flood-fill with downsampling** for efficient ROI separation (64× speedup)
- **Andrew's Monotone Chain** for convex hull computation
- **Douglas-Peucker** for corner extraction with adaptive epsilon
- **SVD-based homography** computation without OpenCV
- **Kalman filter** implementation with outlier rejection and rigid constraints
- **Homography decomposition** for camera pose estimation
- **Depth smoothing** with exponential moving average

---

## Project Structure

```
col780-AR_tags_analysus/
├── README.md                          # This file
├── CAMERA_CALIBRATION.md              # Camera calibration guide
├── Report/                            # Assignment report (LaTeX)
│   ├── main.tex                       # Main report document
│   ├── tau.bib                        # References
│   ├── tau-class/                     # LaTeX template
│   └── figures/                       # Generated visualizations
├── src/                               # Source code
│   ├── main.py                        # Main entry point
│   ├── smoothening_utils.py           # Core AR pipeline with Kalman filtering
│   ├── manual_optimised.py            # Custom CV algorithms (convex hull, etc.)
│   ├── calibrate_camera.py            # Webcam calibration utility
│   ├── generate_aruco.py              # Tag generation scripts
│   ├── generate_custom_artag.py
│   ├── visualize_tag.py
│   └── visualize_ar_steps.py          # Pipeline visualization
├── assets/                            # Input data
│   ├── videos_and_intrinsics/
│   │   ├── Tag0.mp4                   # Single tag video
│   │   ├── multipleTags.mp4           # Multiple tags video
│   │   ├── intrinsics.txt             # Camera intrinsics for videos
│   │   └── webcam_intrinsics.npz      # (Generated) Webcam calibration
│   ├── iitd_logo_template.jpg         # Template for 2D overlay
│   ├── model2.obj, model3.obj         # 3D models for rendering
│   └── *.png                          # AR tag images for printing
├── results/                           # Output videos
│   └── last_result.mp4                # Most recent execution output
└── run.sh                             # Example run commands

```

---

## Environment Setup

### Prerequisites
- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Hardware**: Webcam (optional, for live demo)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/shahi-dKhan/col780-AR_tags_analysus.git
cd col780-AR_tags_analysus
```

2. **Create a virtual environment** (recommended):
```bash
# Using conda (recommended)
conda create -n ar_tags python=3.10
conda activate ar_tags

# OR using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install numpy opencv-python numba matplotlib
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20, <2.0 | Array operations, linear algebra |
| `opencv-python` | ≥4.5 | Video I/O, basic image operations |
| `numba` | ≥0.55 | JIT compilation for speed |
| `matplotlib` | ≥3.3 | (Optional) Visualization |

**Important**: Numba requires NumPy <2.0. If you have NumPy 2.x, downgrade:
```bash
pip install "numpy<2.0"
```

---

## Usage Instructions

The main script `src/main.py` supports three modes:

### Mode 1: Tag Detection Only (Mark Corners)
Detects AR tags and draws bounding boxes with corner markers.

```bash
python src/main.py --video assets/videos_and_intrinsics/multipleTags.mp4
```

**Output:** Displays detected tags with colored corner markers (red, green, blue, yellow) and tag IDs.

### Mode 2: 2D Augmented Reality (Image Overlay)
Overlays a template image on detected AR tags with perspective correction.

```bash
python src/main.py \
    --video assets/videos_and_intrinsics/Tag0.mp4 \
    --template assets/iitd_logo_template.jpg
```

**Output:** Template image warped onto detected tag with Kalman smoothing.

### Mode 3: 3D Augmented Reality (3D Model Rendering)
Renders a 3D .obj model on detected tags using pose estimation.

**For video files:**
```bash
python src/main.py \
    --video assets/videos_and_intrinsics/Tag0.mp4 \
    --intrinsics assets/videos_and_intrinsics/intrinsics.txt \
    --model assets/model2.obj \
    --scale 50
```

**For webcam (requires calibration first):**
```bash
# Step 1: Calibrate webcam (one-time setup)
python src/calibrate_camera.py --mode both --num-images 20

# Step 2: Run 3D AR with webcam
python src/main.py \
    --intrinsics assets/videos_and_intrinsics/webcam_intrinsics.npz \
    --model assets/model3.obj \
    --scale 0.2
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | `None` (webcam) | Path to input video file |
| `--template` | str | `None` | Path to template image for 2D overlay |
| `--model` | str | `None` | Path to .obj file for 3D rendering |
| `--intrinsics` | str | `intrinsics.txt` | Camera calibration file (.txt or .npz) |
| `--scale` | float | `50.0` | Scale factor for 3D model size |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |

---

## Input/Output Specifications

### Input Formats

#### Video Input
- **Format**: MP4, AVI, or any OpenCV-supported video codec
- **Resolution**: Tested on 1920×1080 (1080p)
- **Frame Rate**: 30 FPS recommended
- **Webcam**: Device ID 0 (default camera)

#### Template Image (2D Overlay)
- **Format**: JPG, PNG
- **Recommended Size**: 500×500 to 1000×1000 pixels
- **Aspect Ratio**: Any (will be warped to tag dimensions)

#### 3D Model (3D Rendering)
- **Format**: Wavefront OBJ (.obj)
- **Requirements**: Must contain vertex (`v`) and face (`f`) definitions
- **Coordinate System**: Right-handed, Y-up
- **Scale**: Adjust using `--scale` parameter

#### Camera Intrinsics
- **Format 1**: `.txt` file with NumPy array definition (see `intrinsics.txt`)
- **Format 2**: `.npz` file with keys `['K', 'dist']` (camera calibration output)
- **Contents**: 3×3 intrinsic matrix K with focal lengths and principal point

**Example intrinsics.txt:**
```python
K = np.array([[1406.08, 2.21,  1014.14],
              [0,       1418.0, 566.35],
              [0,       0,      1]])
```

### Output Formats

#### Video Output
- **Path**: `results/last_result.mp4`
- **Codec**: MP4V
- **Resolution**: Same as input
- **Frame Rate**: Same as input
- **Overwrite**: Yes (replaces previous output)

#### Console Output
```
Using video file (calibration not needed for 2D modes)
Saving output to: results/last_result.mp4
Frame 0 | FPS: 11.2 | Tags: 1
Frame 1 | FPS: 11.8 | Tags: 1
...
```

---

## Demo Instructions

### For TAs: Live Demo Setup

1. **Navigate to project directory:**
```bash
cd col780-AR_tags_analysus
```

2. **Activate environment:**
```bash
conda activate ar_tags  # or: source .venv/bin/activate
```

3. **Test with provided videos:**

**Detection demo (multiple tags):**
```bash
python src/main.py --video assets/videos_and_intrinsics/multipleTags.mp4
```

**2D overlay demo:**
```bash
python src/main.py \
    --video assets/videos_and_intrinsics/Tag0.mp4 \
    --template assets/iitd_logo_template.jpg
```

**3D rendering demo:**
```bash
python src/main.py \
    --video assets/videos_and_intrinsics/Tag0.mp4 \
    --intrinsics assets/videos_and_intrinsics/intrinsics.txt \
    --model assets/model2.obj \
    --scale 50
```

### Webcam Demo (Optional)

If webcam is available:

1. **Calibrate camera (one-time):**
```bash
python src/calibrate_camera.py --mode both --num-images 20
# Follow on-screen instructions (press SPACE to capture, 'q' when done)
```

2. **Run webcam demo:**
```bash
# Tag detection
python src/main.py

# 3D rendering (after calibration)
python src/main.py --model assets/model3.obj --scale 0.2
```

### Troubleshooting

**Issue**: "Error opening source"
- **Solution**: Check video path, try absolute path

**Issue**: NumPy version error
- **Solution**: `pip install "numpy<2.0"`

**Issue**: Low FPS (<5 FPS)
- **Possible causes**: High resolution video, slow CPU, Python interpreter overhead
- **Solution**: Ensure Numba is installed, try smaller video resolution

**Issue**: 3D model not rendering
- **Solution**: Verify intrinsics file exists and camera is calibrated

---

## Implementation Details

### Algorithm Pipeline

**Detection (Section 1):**
1. Grayscale conversion + fixed thresholding (T=200)
2. Flood-fill ROI separation with 8× downsampling
3. Fast component labeling (stack-based DFS)
4. Boundary extraction via erosion
5. Convex hull (Andrew's algorithm)
6. Douglas-Peucker approximation (adaptive ε)
7. Homography-based grid decoding (SVD)
8. Orientation and ID extraction

**2D Overlay (Section 2):**
1. Tag detection (above pipeline)
2. Kalman filtering (4-state per corner)
3. Homography computation
4. Perspective warping of template
5. Mask generation and blending

**3D Rendering (Section 3):**
1. Tag detection with Kalman filtering
2. Homography decomposition → R, t
3. SVD orthogonalization of rotation matrix
4. Projection matrix P = K[R|t]
5. 3D→2D projection of vertices
6. Face rasterization with depth smoothing

### Optimization Techniques

| Technique | Speedup | Description |
|-----------|---------|-------------|
| Downsampling | 64× | ROI separation at 1/8 resolution |
| NumPy vectorization | 10-20× | Replace Python loops with array ops |
| Numba JIT | 5-10× | Compile critical functions to machine code |
| Kalman filtering | N/A | Smooth+predict → reduce false detections |
| Stack-based DFS | 2× | Avoid recursion limits |

**Performance trajectory:**
- Initial implementation: 0.2 FPS
- After NumPy optimization: 2-3 FPS
- After downsampling: 8-10 FPS
- Final (with Numba): 11-13 FPS

---

## Performance

### Benchmarks (1920×1080 video, MacBook Pro M1)

| Mode | FPS | Latency | Tags/Frame |
|------|-----|---------|------------|
| Detection only | 12-13 | 77ms | 1-3 |
| 2D Overlay | 11-12 | 85ms | 1 |
| 3D Rendering | 10-12 | 90ms | 1 |

### Accuracy
- **Detection Rate**: >95% for tags at 30-200cm distance
- **Orientation Accuracy**: ~95% (occasional flips at extreme angles)
- **Jitter Reduction**: ~70% with Kalman filtering

---

## Known Limitations

1. **Brightness Sensitivity**: Fixed threshold (T=200) fails on very bright/dark backgrounds
2. **Motion Blur**: Fast camera movement causes detection failures
3. **Orientation Flips**: Noise at tag edges causes incorrect rotation detection (~5% error rate)
4. **Single Tag for 3D**: 3D rendering optimized for single large tag, multiple tags may cause flickering
5. **Perspective Limits**: Detection fails beyond ~60° viewing angle
6. **Lighting**: Requires reasonably uniform lighting (no harsh shadows on tag)

### Failure Cases
- Overexposed/underexposed video
- Extreme perspective distortion (>70° angle)
- Partial tag occlusion
- Tags smaller than ~100×100 pixels in frame

---

## Additional Resources

- **Report**: See `Report/main.tex` for detailed algorithm descriptions and mathematical formulations
- **Calibration Guide**: See `CAMERA_CALIBRATION.md` for webcam setup instructions
- **Video Results**: Check `results/` directory for output from each run
- **Generated Figures**: See `Report/figures/` for pipeline visualizations

---

## Citation

If you use this code for academic purposes, please cite:

```
@misc{khan2026artags,
  author = {Shahid Khan},
  title = {COL780 AR Tags Analysis and Augmented Reality},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shahi-dKhan/col780-AR_tags_analysus}}
}
```

---

## License

This project is submitted as part of COL780 course assignment at IIT Delhi.  
Academic use only. Not licensed for commercial use.

---

## Contact

**Shahid Khan**  
Email: ee1221163@iitd.ac.in  
Course: COL780 - Computer Vision  
Institution: Indian Institute of Technology Delhi

---

**Last Updated:** February 15, 2026

