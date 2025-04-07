# 002

import cv2
import numpy as np

def order_points(pts):
    """Orders the corner points in a specific order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="int32")
    points = np.array(pts)
    sorted_points = points[np.argsort(points[:, 1])]
    y_coords = sorted_points[:, 1]
    y_diffs = np.diff(y_coords)
    threshold_index = np.argmax(y_diffs)
    split_value = y_coords[threshold_index]
    top_points = sorted_points[sorted_points[:, 1] <= split_value]
    bottom_points = sorted_points[sorted_points[:, 1] > split_value]
    top_left = min(top_points, key=lambda p: p[0])
    top_right = max(top_points, key=lambda p: p[0])
    bottom_left = min(bottom_points, key=lambda p: p[0])
    bottom_right = max(bottom_points, key=lambda p: p[0])
    rect[0], rect[1], rect[2], rect[3] = top_left, top_right, bottom_right, bottom_left
    return rect

def find_floor_contour(mask_path):
    """Finds the largest contour in the given binary mask image."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    approx = cv2.convexHull(largest_contour)
    return approx.reshape(-1, 2), binary_mask

def apply_homography(tile_img, ordered_corners, mask_shape):
    """Applies homography to warp the tile image onto the detected floor area."""
    tile_h, tile_w = tile_img.shape[:2]
    src_pts = np.array([[0, 0], [tile_w, 0], [tile_w, tile_h], [0, tile_h]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, ordered_corners)
    return cv2.warpPerspective(tile_img, H, (mask_shape[1], mask_shape[0]))

def overlay_texture_on_floor(original_image, mask_path, tile_path):
    """Overlays a tile texture onto the detected floor area of an image."""
    corners, binary_mask = find_floor_contour(mask_path)
    if corners is None:
        return
    ordered_corners = order_points(corners)
    original_image = cv2.imread(original_image)
    tile = cv2.imread(tile_path)
    tiled_image = np.tile(tile, (2, 2, 1))
    warped_tile = apply_homography(tiled_image, ordered_corners, binary_mask.shape)
    carpet_mask = cv2.bitwise_not(cv2.cvtColor(warped_tile, cv2.COLOR_BGR2GRAY))
    uncovered_mask = cv2.bitwise_and(binary_mask, cv2.threshold(carpet_mask, 250, 255, cv2.THRESH_BINARY)[1])
    resized_mask = cv2.resize(tiled_image, (uncovered_mask.shape[1], uncovered_mask.shape[0]))
    tresult = np.where(uncovered_mask[:, :, None] == 255, resized_mask, warped_tile)
    final_result = np.where(binary_mask[:, :, None] == 255, tresult, original_image)
    return final_result

def main():
    mask_path = "D:/Quleep/Prototype/Code/mask_output/demo1.jpg"
    tile_path = "D:/Quleep/Prototype/Code/Data/floor4.jpg"
    original_image_path = "D:/Quleep/Prototype/Code/Data/image1.jpg"
    final_result = overlay_texture_on_floor(original_image_path, mask_path, tile_path)
    if final_result is not None:
        cv2.imshow("Final Warped Tile", final_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("D:/Quleep/Prototype/Code/Data/output/final_result.jpg", final_result)

if __name__ == "__main__":
    main()
