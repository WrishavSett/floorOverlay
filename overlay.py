# 015

import os
import cv2
import numpy as np
from scale_and_overlay import place_on_black
from convert_binary import convert_to_binary_mask, convert_to_binary_carpet
from carpet_circle import carpet_ellipse_and_center

def adjust_carpet_perspective(carpet_img_path, temp_path="../floorOverlay/temporary"):
    image = cv2.imread(carpet_img_path)
    h, w = image.shape[:2]

    # Define the source points (corners of the original image)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Compute the new upper width to form a 135-degree trapezoid
    # offset = h // 2  # This defines how much the top should be shrunk
    # new_w_top = w - 2 * offset  # Ensure both sides shrink equally

    # Reduce the offset to achieve 110 degrees (less shrinking)
    offset = h // 3  # Reduced from h // 2 to create a wider top

    # Compute new upper width
    new_w_top = w - 2 * offset

    # Define the destination points for the new perspective
    dst_pts = np.float32([
        [offset, 0],         # Top-left (shifted inward slightly)
        [w - offset, 0],     # Top-right (shifted inward slightly)
        [w, h],              # Bottom-right (unchanged)
        [0, h]               # Bottom-left (unchanged)
    ])

    # Compute perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, matrix, (w, h))

    warped_img_path = os.path.join(temp_path, "warped_carpet_image.jpg")
    cv2.imwrite(warped_img_path, warped)

    return warped_img_path

def overlay_carpet_trapezoid(room_img_path, carpet_img_path, output_path="../floorOverlay/final_out"):
    warped_carpet_img_path = adjust_carpet_perspective(carpet_img_path)
    room_img = cv2.imread(room_img_path)
    # Extract the room image name without extension
    room_image_name = os.path.splitext(os.path.basename(room_img_path))[0]

    # mask_img_path = mask(room_img_path)
    # mask_img = cv2.imread(mask_img_path)

    bin_mask_img_path = convert_to_binary_mask(room_img_path)
    bin_mask_img = cv2.imread(bin_mask_img_path)

    overlayed_carpet_img_path = place_on_black(room_img_path, warped_carpet_img_path)
    overlayed_carpet_img = cv2.imread(overlayed_carpet_img_path)

    overlayed_bin_carpet_img_path = convert_to_binary_carpet(overlayed_carpet_img_path)
    overlayed_bin_carpet_img = cv2.imread(overlayed_bin_carpet_img_path)

    # Bitwise AND between binary masked room image and overlayed carpet image
    tmp_result = cv2.bitwise_and(bin_mask_img, overlayed_bin_carpet_img)
    result = np.where(tmp_result == 255, overlayed_carpet_img,room_img)

    
    result_img_path = os.path.join(output_path, f"overlayed_carpet_t_{room_image_name}.jpg")
    cv2.imwrite(result_img_path, result)

    return result_img_path

def overlay_carpet_ellipse(room_img_path, carpet_img_path, output_path="../floorOverlay/final_out"):
    ellipse_carpet_path, ellipse_carpet_center = carpet_ellipse_and_center(carpet_img_path)
    room_img = cv2.imread(room_img_path)
    # Extract the room image name without extension
    room_image_name = os.path.splitext(os.path.basename(room_img_path))[0]

    # mask_img_path = mask(room_img_path)
    # mask_img = cv2.imread(mask_img_path)

    bin_mask_img_path = convert_to_binary_mask(room_img_path)
    bin_mask_img = cv2.imread(bin_mask_img_path)

    overlayed_carpet_img_path = place_on_black(room_img_path, ellipse_carpet_path)
    overlayed_carpet_img = cv2.imread(overlayed_carpet_img_path)

    overlayed_bin_carpet_img_path = convert_to_binary_carpet(overlayed_carpet_img_path)
    overlayed_bin_carpet_img = cv2.imread(overlayed_bin_carpet_img_path)

    # Bitwise AND between binary masked room image and overlayed carpet image
    tmp_result = cv2.bitwise_and(bin_mask_img, overlayed_bin_carpet_img)
    result = np.where(tmp_result == 255, overlayed_carpet_img,room_img)

    
    result_img_path = os.path.join(output_path, f"overlayed_carpet_e_{room_image_name}.jpg")
    cv2.imwrite(result_img_path, result)

    return result_img_path

def main():
    room_img_path = "../floorOverlay/inputRoom/room6.jpg"
    carpet_img_path = "../floorOverlay/inputCarpet/carpet2.jpg"
    temp_folder_path = "../floorOverlay/temporary"
    
    overlay_carpet_trapezoid(room_img_path, carpet_img_path)
    overlay_carpet_ellipse(room_img_path, carpet_img_path)

if __name__ == "__main__":
    main()