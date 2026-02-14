import cv2 # OpenCV - for image read/write, video capture, and display
import argparse
import time
import os
from smoothening_utils import * # Define custom CV functions in utils.py
# from utils_optimised import *
def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    parser.add_argument("--intrinsics", type=str, help="Path to camera intrinsics file.", 
                        default="assets/videos_and_intrinsics/intrinsics.txt")
    parser.add_argument("--scale", type=float, help="Scale factor for 3D model.", default=50.0)

    args = parser.parse_args()
    template_image = cv2.imread(args.template) if args.template else None
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    fps_for_dt = float(video_fps) if video_fps and video_fps > 1e-3 else 30.0
    
    # Create Kalman trackers for smoothing - Lower process noise and higher measurement noise = more smoothing
    kalman_tracker_3d = KalmanTagTracker(process_noise=0.5, measurement_noise=12.0, max_innovation=150)
    kalman_tracker_2d = KalmanTagTracker(process_noise=0.5, measurement_noise=12.0, max_innovation=150)
    kalman_tracker_marking = KalmanTagTracker(process_noise=0.8, measurement_noise=10.0, max_innovation=150)
    
    # Setup video writer - always save output
    os.makedirs("results", exist_ok=True)
    output_path = "results/last_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))
    print(f"Saving output to: {output_path}")

    frame_count = 0
    fps = 0.0
    window_name = "Frame"
    
    while cap.isOpened():
        start_time = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame based on mode
        if args.model:
            # 3D model rendering
            frame = process_frame_3D(frame, args.model, args.intrinsics, scale_3d=args.scale, scale=4, smoother=kalman_tracker_3d)
        elif args.template:
            # 2D template superimposition
            frame = process_frame_superimpose(frame, args.template, scale=4, smoother=kalman_tracker_2d)
        else:
            # Just mark corners
            frame = process_frame_marking(frame, scale=4, smoother=kalman_tracker_marking)

        end_time = time.perf_counter()
        dt = end_time - start_time

        if dt > 0:
            inst_fps = 1.0 / dt
            fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Write to output video
        out.write(frame)
        print(f"Processing frame {frame_count}...", end='\r')

        # Scale display to 50% for better screen fit
        display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), 27):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    out.release()
    print(f"\nVideo saved successfully to results/last_result.mp4! Total frames: {frame_count}")
    cv2.destroyAllWindows()


# 

if __name__ == "__main__":
    main()


