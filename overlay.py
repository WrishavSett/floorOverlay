# 015

import os
import cv2
import numpy as np
from mask_room_image import mask
from scale_and_overlay import place_on_black, create_black_image
from convert_binary import convert_to_binary_mask, convert_to_binary_carpet
from carpet_circle import carpet_ellipse_and_center
from find_centroid import find_and_mark_floor_center

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

    
    result_img_path = os.path.join(output_path, f"overlayed_carpet_{room_image_name}.jpg")
    cv2.imwrite(result_img_path, result)

    return result_img_path

# def overlay_carpet_ellipse(room_img_path, carpet_img_path, output_path="../floorOverlay/final_out"):
#     room_image_name = os.path.splitext(os.path.basename(room_img_path))[0]

#     # Step 1: Detect center of floor (red mask) in the room image
#     floor_center = find_and_mark_floor_center(room_img_path)
#     if not floor_center:
#         print("015 Could not detect floor center in room image.")
#         return

#     fx, fy = floor_center  # Floor center coordinates in room image

#     # Step 2: Create a blank black background the same size as the room
#     black_bg_path = create_black_image(room_img_path)
#     black_bg = cv2.imread(black_bg_path)

#     # Step 3: Process carpet image to get ellipse and its center
#     carpet_path, carpet_center = carpet_ellipse_and_center(carpet_img_path)
#     carpet = cv2.imread(carpet_path, cv2.IMREAD_UNCHANGED)
#     cx, cy = carpet_center  # Center of the ellipse in the carpet image

#     # Step 4: Align carpet ellipse center with floor center
#     top_left_x = fx - cx
#     top_left_y = fy - cy

#     h, w = carpet.shape[:2]
#     overlay = black_bg.copy()

#     # Step 5: Ensure the carpet doesn't go out of bounds
#     x1 = max(top_left_x, 0)
#     y1 = max(top_left_y, 0)
#     x2 = min(top_left_x + w, overlay.shape[1])
#     y2 = min(top_left_y + h, overlay.shape[0])

#     crop_x1 = x1 - top_left_x
#     crop_y1 = y1 - top_left_y
#     crop_x2 = crop_x1 + (x2 - x1)
#     crop_y2 = crop_y1 + (y2 - y1)

#     if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
#         print("015 Invalid crop dimensions for overlay. Aborting.")
#         return

#     # Step 6: Crop carpet and background regions
#     roi = overlay[y1:y2, x1:x2]
#     carpet_crop = carpet[crop_y1:crop_y2, crop_x1:crop_x2]

#     # Step 7: Alpha blending
#     if carpet_crop.shape[2] == 4:
#         b, g, r, a = cv2.split(carpet_crop)
#         carpet_rgb = cv2.merge((b, g, r))
#         alpha_mask = cv2.merge((a, a, a)) / 255.0
#     else:
#         carpet_rgb = carpet_crop
#         alpha_mask = np.ones_like(carpet_rgb, dtype=np.float32)

#     blended = (roi * (1 - alpha_mask) + carpet_rgb * alpha_mask).astype(np.uint8)
#     overlay[y1:y2, x1:x2] = blended

#     # Step 8: Overlay the result onto the actual room image
#     room_img = cv2.imread(room_img_path)
#     final_output = cv2.addWeighted(room_img, 1.0, overlay, 1.0, 0)

#     final_output_path = os.path.join(output_path, f"overlayed_carpet_ellipse_{room_image_name}.jpg")
#     cv2.imwrite(final_output_path, final_output)
#     print(f"015 Final carpet overlay saved to: {final_output_path}")

def main():
    room_img_path = "../floorOverlay/inputRoom/room4.jpg"
    carpet_img_path = "../floorOverlay/inputCarpet/carpet2.jpg"
    temp_folder_path = "../floorOverlay/temporary"
    
    overlay_carpet_trapezoid(room_img_path, carpet_img_path)
    # overlay_carpet_ellipse(room_img_path, carpet_img_path)

if __name__ == "__main__":
    main()