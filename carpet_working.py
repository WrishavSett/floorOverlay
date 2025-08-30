# 002

import cv2
import numpy as np

def order_points(pts):
    """
    Orders the corner points of a quadrilateral in a consistent manner (top-left, top-right, bottom-right, bottom-left).

    Args:
        pts (np.array): A NumPy array of shape (N, 2) representing N points.

    Returns:
        np.array: A NumPy array of shape (4, 2) containing the ordered corner points.
    """
    rect = np.zeros((4, 2), dtype="int32")
    points = np.array(pts)
    sorted_points = points[np.argsort(points[:, 1])]
    y_coords = sorted_points[:, 1]
    y_diffs = np.diff(y_coords)
    threshold_index = np.argmax(y_diffs)
    split_value = y_coords[threshold_index]
    top_points = sorted_points[sorted_points[:, 1] <= split_value]
    bottom_points = sorted_points[sorted_points[:, 1] > split_value]
    min_y_index = np.argmin(top_points[:, 1])
    lowest_y_point = top_points[min_y_index]
    top_left = min(top_points, key=lambda p: p[0])
    top_right = max(top_points, key=lambda p: p[0])
    top_left = np.array(top_left)
    top_right = np.array(top_right)
    top_left[1] = lowest_y_point[1]
    top_right[1] = lowest_y_point[1]
    max_y_index = np.argmax(bottom_points[:, 1])
    highest_y_point = bottom_points[max_y_index]
    bottom_left = min(bottom_points, key=lambda p: p[0])
    bottom_right = max(bottom_points, key=lambda p: p[0])
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)
    bottom_left[1] = highest_y_point[1]
    bottom_right[1] = highest_y_point[1]
    rect[0], rect[1], rect[2], rect[3] = top_left, top_right, bottom_right, bottom_left
    return rect

def find_floor_contour(mask_path):
    """
    Finds the largest contour representing the floor in a given binary mask image.

    Args:
        mask_path (str): The file path to the binary mask image.

    Returns:
        tuple: A tuple containing the approximated contour points (np.array) and the binary mask (np.array), or None if no contours are found.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("|ERROR| No contours found in the provided mask image.")
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    approx = cv2.convexHull(largest_contour)
    return approx.reshape(-1, 2), binary_mask

def apply_homography(tile_img, ordered_corners, mask_shape):
    """
    Applies homography to warp a tile image onto a quadrilateral defined by ordered corners.

    Args:
        tile_img (np.array): The source tile image.
        ordered_corners (np.array): An array of the four ordered corner points of the destination area.
        mask_shape (tuple): The shape (height, width) of the target mask or image.

    Returns:
        np.array: The warped image.
    """
    tile_h, tile_w = tile_img.shape[:2]
    src_pts = np.array([[0, 0], [tile_w, 0], [tile_w, tile_h], [0, tile_h]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, ordered_corners)
    return cv2.warpPerspective(tile_img, H, (mask_shape[1], mask_shape[0]))

def overlay_texture_on_floor(original_image_path, mask_path, tile_path):
    """
    Overlays a tile texture onto the floor area detected in a mask.

    Args:
        original_image_path (str): The path to the original image.
        mask_path (str): The path to the segmentation mask of the floor.
        tile_path (str): The path to the tile texture image.

    Returns:
        np.array: The final image with the floor texture overlaid, or None if the process fails.
    """
    corners, binary_mask = find_floor_contour(mask_path)
    if corners is None:
        return None
    ordered_corners = order_points(corners)
    original_image = cv2.imread(original_image_path)
    tile = cv2.imread(tile_path)
    if original_image is None or tile is None:
        print("|ERROR| Could not read one of the input images. Please check the paths.")
        return None
    
    tiled_image = np.tile(tile, (2, 2, 1))
    warped_tile = apply_homography(tiled_image, ordered_corners, binary_mask.shape)
    
    carpet_mask = cv2.bitwise_not(cv2.cvtColor(warped_tile, cv2.COLOR_BGR2GRAY))
    uncovered_mask = cv2.bitwise_and(binary_mask, cv2.threshold(carpet_mask, 250, 255, cv2.THRESH_BINARY)[1])
    resized_mask = cv2.resize(tiled_image, (uncovered_mask.shape[1], uncovered_mask.shape[0]))
    tresult = np.where(uncovered_mask[:, :, None] == 255, resized_mask, warped_tile)
    final_result = np.where(binary_mask[:, :, None] == 255, tresult, original_image)
    return final_result

def main():
    """
    Main function to demonstrate the process of overlaying a tile texture on a floor.
    It defines file paths, calls the overlay function, and handles the output.
    """
    mask_path = "D:/Quleep/Prototype/Code/mask_output/demo1.jpg"
    tile_path = "D:/Quleep/Prototype/Code/Data/floor4.jpg"
    original_image_path = "D:/Quleep/Prototype/Code/Data/image1.jpg"
    
    print(f"|INFO| Starting texture overlay process for: {original_image_path}")
    final_result = overlay_texture_on_floor(original_image_path, mask_path, tile_path)
    
    if final_result is not None:
        output_path = "D:/Quleep/Prototype/Code/Data/output/final_result.jpg"
        cv2.imwrite(output_path, final_result)
        print(f"|OUTPUT| Final image successfully saved to: {output_path}")
        # Optional: Display the image
        # cv2.imshow("Final Warped Tile", final_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print(f"|ERROR| The texture overlay process failed for the image: {original_image_path}")

if __name__ == "__main__":
    main()