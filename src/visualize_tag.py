import cv2
from utils_optimised import generate_tag

if __name__ == "__main__":
    tag_id = 15
    tag_img = generate_tag(tag_id)
    cv2.imshow(f"AR Tag ID {tag_id}", tag_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    