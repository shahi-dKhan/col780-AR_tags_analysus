import cv2 # OpenCV - for image read/write, video capture, and display
import argparse
import time
from utils import * # Define custom CV functions in utils.py

def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)

    args = parser.parse_args()
    template_image = cv2.imread(args.template) if args.template else None
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return

    prev_t = time.perf_counter()
    fps = 0.0

    window_name = "Frame"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)

        now_t = time.perf_counter()
        dt = now_t - prev_t
        prev_t = now_t
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

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), 27):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


# 

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Original cv2-based main loop (commented out per request)
# -----------------------------------------------------------------------------


# def main():
#     parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
#     parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
#     parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
#     parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    
#     args = parser.parse_args()
#     template_image = cv2.imread(args.template) if args.template else None
#     video_source = args.video if args.video else 0
#     cap = cv2.VideoCapture(video_source)
    
#     if not cap.isOpened():
#         print(f"Error opening source: {video_source}")
#         return

#     prev_t = time.perf_counter()
#     fps = 0.0
        
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # gray = threshold_image(frame)
#         # cv2.imshow("Thresholded", gray)
#         # islands = split_ROI(gray, min_sheet_area=2000)
#         # print("len(islands):", len(islands))
#         # marker_pixels = []
#         # for i in range(len(islands)):
#         #     marker_pixel = detect_tag(frame, islands[i], gray)
#         #     if marker_pixel is not None:
#         #         tag = decode_tag(marker_pixel, gray)
#         #         if tag:
#         #             print(f"Detected Tag ID: {tag.id}")
#         #             mark_corners(frame, tag)
#                     # frame = superimpose(frame, template_image, tag=tag)
#         # make a image where only the marker pixel is white and the rest is black
#         # print("Length of marker", len(marker_pixel))
#         frame = process_frame(frame)

#         # FPS (exponential moving average to reduce jitter)
#         now_t = time.perf_counter()
#         dt = now_t - prev_t
#         prev_t = now_t
#         if dt > 0:
#             inst_fps = 1.0 / dt
#             fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

#         cv2.putText(
#             frame,
#             f"FPS: {fps:.1f}",
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1.0,
#             (0, 255, 0),
#             2,
#             cv2.LINE_AA,
#         )

#         cv2.imshow("Frame", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()