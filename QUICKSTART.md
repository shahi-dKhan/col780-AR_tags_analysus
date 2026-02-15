# Quick Start Guide - AR Tags Demo

## 5-Minute Setup

1. **Install dependencies:**
```bash
conda create -n ar_tags python=3.10 -y
conda activate ar_tags
pip install -r requirements.txt
```

2. **Run demos:**

### Demo 1: AR Tag Detection (Multiple Tags)
```bash
python src/main.py --video assets/videos_and_intrinsics/multipleTags.mp4
```
**What to expect:** Detects multiple AR tags, shows bounding boxes with corners in different colors, displays tag IDs

### Demo 2: 2D Image Overlay
```bash
python src/main.py --video assets/videos_and_intrinsics/Tag0.mp4 --template assets/iitd_logo_template.jpg
```
**What to expect:** IITD logo overlaid on AR tag with perspective correction and Kalman smoothing

### Demo 3: 3D Model Rendering
```bash
python src/main.py --video assets/videos_and_intrinsics/Tag0.mp4 --intrinsics assets/videos_and_intrinsics/intrinsics.txt --model assets/model2.obj --scale 50
```
**What to expect:** 3D model rendered on AR tag with pose estimation

### Demo 4 (Optional): Webcam Mode
```bash
# First-time setup: calibrate your webcam
python src/calibrate_camera.py --mode both --num-images 20
# Press SPACE to capture images, 'q' when done

# Then run with webcam
python src/main.py --model assets/model3.obj --scale 0.2
```

## Output Location
All results are saved to: `results/last_result.mp4`

## Keyboard Controls
- `q` or `ESC`: Quit
- `SPACE`: Pause/Resume

## Quick Troubleshooting
- **NumPy error**: Run `pip install "numpy<2.0"`
- **Video won't play**: Check file path or try absolute path
- **Low FPS**: Normal, ~12 FPS expected for 1080p video

## File Structure for Demo
```
col780-AR_tags_analysus/
├── src/main.py              # Main script
├── assets/
│   ├── videos_and_intrinsics/
│   │   ├── Tag0.mp4         # Single tag video
│   │   └── multipleTags.mp4 # Multiple tags
│   ├── iitd_logo_template.jpg
│   └── model2.obj, model3.obj
└── results/
    └── last_result.mp4      # Auto-generated output
```

## For Detailed Information
See `README.md` for complete documentation
