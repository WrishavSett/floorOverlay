# Redundant as of now
# Computational code for floor overlaying

import os
import cv2
import numpy as np
import pandas as pd
from floor_mask_model import infer


#021
def base_image(room_img_path, height_mul, width_mul , temp="../floorOverlay/temporary"):
    """
    room_img_path: path to the room image
    height_mul: height multiplier
    width_mul: width multiplier
    returns: path to the base image
    """
    # Load the room image
    room = cv2.imread(room_img_path)

    if room is None:
        raise FileNotFoundError(f"021 Image at path {room_img_path} not found.")

    # Get the original dimensions of the room image
    room_height = room.shape[0]
    room_width = room.shape[1]

    # Print the size of the input room image
    print(f"021 Original room image size: {room_height}x{room_width}")

    # Create a base image with the scaled dimensions
    base_image = np.zeros((int(height_mul * room_height), int(width_mul * room_width), 3), dtype=np.uint8)

    # Print the size of the saved base image
    print(f"021 Base image size: {base_image.shape[0]}x{base_image.shape[1]}")

    base_img_path = f"{temp}/base_img.jpg"
    # Save the base image
    cv2.imwrite(base_img_path, base_image)
    print(f"021 Base image saved at {temp}")

    return base_img_path


#026
def tiling(floor_img_path, repeat_y, repeat_x, temp="../floorOverlay/temporary"):
    tile_img = cv2.imread(floor_img_path)

    if tile_img is None:
        raise ValueError(f"Failed to read image at path: {floor_img_path}")
    
    resized = cv2.resize(tile_img, (512, 512), interpolation=cv2.INTER_AREA)

    tiled_output_path = f"{temp}/tiled_floor.jpg"
    repeated = np.tile(resized, (repeat_y, repeat_x, 1))
    cv2.imwrite(tiled_output_path, repeated)
    print(f"026 Tiled floor image saved at {temp}")

    return tiled_output_path


#022
def floor_prep(room_img_path, floor_img_path, height_mul, width_mul, temp="../floorOverlay/temporary"):
    """
    returns: the image of the tile perspective transformed and placed on the base image from 021
    """

    base_img_path = base_image(room_img_path, height_mul, width_mul)
    base_img = cv2.imread(base_img_path)
    base_height, base_width = base_img.shape[:2]

    tiled_floor_img_path = tiling(floor_img_path, 10, 8)

    # Load tile image
    tile = cv2.imread(tiled_floor_img_path)
    if tile is None:
        raise FileNotFoundError(f"022 Image at path {floor_img_path} not found.")

    tile_height, tile_width = tile.shape[:2]

    # Source points: corners of the tile image
    src_pts = np.float32([
        [0, 0],                    # top-left
        [tile_width - 1, 0],       # top-right
        [0, tile_height - 1],      # bottom-left
        [tile_width - 1, tile_height - 1]  # bottom-right
    ])

    # Destination points on base image
    dst_pts = np.float32([
        [base_width // 3, 0],                     # top-left → top 1/3
        [2 * base_width // 3, 0],                 # top-right → top 2/3
        [0, base_height - 1],                     # bottom-left
        [base_width - 1, base_height - 1]        # bottom-right
    ])

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the tile image to fit the trapezoid
    warped_tile = cv2.warpPerspective(tile, matrix, (base_width, base_height))

    # Place the warped tile on base image (direct overlay)
    mask = np.any(warped_tile != [0, 0, 0], axis=-1)
    base_img[mask] = warped_tile[mask]

    base_floor_img_path = f"{temp}/base_floor_img.jpg"
    # Save the result
    cv2.imwrite(base_floor_img_path, base_img)
    print(f"022 Base floor image saved at {temp}")

    return base_floor_img_path


#023
def masking(room_img_path, temp="../floorOverlay/mask_out"):
    """
    returns: room mask image
    """
    # load_model()
    room_name = os.path.splitext(os.path.basename(room_img_path))[0]
    mask_output_path = f"{temp}/{room_name}_mask.jpg"
    infer(room_img_path, 0, mask_output_path)
    print(f"023 Masked floor image saved at {temp}")

    return mask_output_path


#024
def crop_image(room_img_path, floor_img_path, height_mul, width_mul, temp="../floorOverlay/temporary"):
    """
    Crops a section of an image based on provided top-left and bottom-right coordinates.

    Args:
        image_path (str): Path to the image file.
        top_left (tuple): (x, y) coordinates of the top-left corner of the cropping region.
        bottom_right (tuple): (x, y) coordinates of the bottom-right corner of the cropping region.

    Returns:
        numpy.ndarray: The cropped image as a NumPy array, or None if the image cannot be loaded or if the coordinates are invalid.
    """

    base_floor_path = floor_prep(room_img_path, floor_img_path, height_mul, width_mul)
    base_floor = cv2.imread(base_floor_path)
    dim = base_floor.shape
    print("024 Dimensions of Base Floor Image: ", dim)

    h = dim[0]
    w = dim[1]
    print("024 Height and Width of Base Floor Image: ", h, w)

    top_left = (w/width_mul, 0)
    bottom_right = (2*w/width_mul, h/height_mul)
    print("024 Top left and Bottom right corner points of the Base Floor Image for Crop: ", top_left, bottom_right)

    try:
        img = cv2.imread(base_floor_path)
        if img is None:
            print(f"024 Error: Could not open or read image file: {base_floor_path}")
            return None

        x1, y1 = top_left
        x2, y2 = bottom_right
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Check if coordinates are valid
        if not (0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0] and 0 <= x2 < img.shape[1] and 0 <= y2 < img.shape[0] and x1 <= x2 and y1 <= y2):
            print("024 Error: Invalid coordinates. Please ensure that coordinates are within the image boundaries and x1 < x2 and y1 < y2.")
            return None

        cropped_img = img[y1:y2, x1:x2]
        cropped_img_path = f"{temp}/cropped_img.jpg"
        cv2.imwrite(cropped_img_path, cropped_img)
        print(f"024 Cropped floor image saved at {temp}")

        return cropped_img_path
    
    except Exception as e:
        print(f"024 An error occurred: {e}")
        return None


#025
def overlay(room_img_path, floor_img_path, height_mul, width_mul, final_out="../floorOverlay/final_out"):
    """
    returns: floor overlayed on mask
    """
    room_mask_path = masking(room_img_path)
    name = os.path.splitext(os.path.basename(room_img_path))[0]
    mask = cv2.imread(room_mask_path)
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh1 = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)

    tile_crop_path = crop_image(room_img_path, floor_img_path, height_mul, width_mul)
    room = cv2.imread(room_img_path)
    tile = cv2.imread(tile_crop_path)
    
    if room is None:
        raise ValueError(f"025 Could not read room image at {room_img_path}")
    if tile is None:
        raise ValueError(f"025 Could not read cropped tile at {tile_crop_path}")
    
    thresh2 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
    result = np.where(thresh2 == 0, room, tile)

    output_path = f"{final_out}/{name}_final_output.jpg"
    cv2.imwrite(output_path, result)
    print(f"025 Final output image saved at {output_path}")
    
    return output_path


def main():
    room_img_path = "../floorOverlay/inputRoom/room1.jpg"
    floor_img_path = "../floorOverlay/inputTile/tile2.jpg"
    height_mul = 2
    width_mul = 3
    overlay(room_img_path, floor_img_path, height_mul, width_mul)


if __name__ == "__main__":
    main()