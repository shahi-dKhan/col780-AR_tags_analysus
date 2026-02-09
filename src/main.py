import cv2 # OpenCV - Only for video capture and display
import argparse
from utils import * # Define custom CV functions in utils.py




def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    
    args = parser.parse_args()
    
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = threshold_image(frame)
        islands = split_ROI(gray, min_sheet_area=500)
        # print(len(islands))
        island = render_ROI(frame, islands, 0)
        marker_pixels = []
        for i in range(len(islands)):
            marker_pixel = detect_tag(frame, islands[i], gray)
            marker_pixels.append(marker_pixel)
        # make a image where only the marker pixel is white and the rest is black
        # print("Length of marker", len(marker_pixel))
        marker_image = np.zeros_like(gray)
        for marker_pixel in marker_pixels:
            for (y, x) in marker_pixel:
                marker_image[y, x] = 255
        cv2.imshow("Frame", marker_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
