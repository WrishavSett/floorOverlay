# 014

from find_centroid import find_and_mark_floor_center
import os
import cv2
import numpy as np

def scale_carpet(room_img_path, carpet_img_path, temp_path="../floorOverlay/temporary"):
    # original_carpet_binary_path = convert_to_binary_carpet(carpet_img_path, temp_path)
    
    ref_image = cv2.imread(room_img_path)
    ref_height, ref_width = ref_image.shape[:2]

    # Compute maximum allowable dimensions (1/5th of reference image)
    # max_width = max(1, ref_width // 5)
    # max_height = max(1, ref_height // 5)
    max_width = max(1, ref_width // 3)
    max_height = max(1, ref_height // 3)

    # Load the input image
    # img = cv2.imread(original_carpet_binary_path)
    img = cv2.imread(carpet_img_path)
    img_height, img_width = img.shape[:2]

    # Compute scaling factor while preserving aspect ratio
    scale_factor = min(max_width / img_width, max_height / img_height)

    # Compute new dimensions
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    # Resize the input image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Ensure the "temporary" folder exists
    output_folder = temp_path
    os.makedirs(output_folder, exist_ok=True)

    # Define output file path
    # scaled_carpet_binary_path = os.path.join(output_folder, "scaled_carpet_binary_image.jpg")
    scaled_carpet_path = os.path.join(output_folder, "scaled_carpet_image.jpg")


    # Save the resized image
    cv2.imwrite(scaled_carpet_path, resized_img)
    print(f"014 Resized image saved as {scaled_carpet_path} with dimensions {new_width}x{new_height}")

    return scaled_carpet_path

def create_black_image(room_img_path, temp_path="../floorOverlay/temporary"):
    # Load the reference image to get its dimensions
    ref_img = cv2.imread(room_img_path)
    ref_height, ref_width = ref_img.shape[:2]

    # Create a black image of the same dimensions
    black_img = np.zeros((ref_height, ref_width, 3), dtype=np.uint8)

    # Ensure the "temporary" folder exists
    output_folder = temp_path
    os.makedirs(output_folder, exist_ok=True)

    # Define output file path
    black_blank_img_path = os.path.join(output_folder, "black_blank_image.jpg")

    # Save the black image
    cv2.imwrite(black_blank_img_path, black_img)
    print(f"014 Black image saved as {black_blank_img_path} with dimensions {ref_width}x{ref_height}")

    return black_blank_img_path

def place_on_black(room_img_path, carpet_img_path, temp_path="../floorOverlay/temporary"):
    center_of_mask = find_and_mark_floor_center(room_img_path, temp_path)
    x, y = center_of_mask
    background_path = create_black_image(room_img_path, temp_path)
    foreground_path = scale_carpet(room_img_path, carpet_img_path, temp_path)

    # Load background (background)
    background = cv2.imread(background_path)
    h2, w2, _ = background.shape

    # Load foreground (foreground)
    foreground = cv2.imread(foreground_path)
    h1, w1, _ = foreground.shape

    # Calculate top-left corner where foreground should be placed
    x1_start = x - w1 // 2
    y1_start = y - h1 // 2

    # Ensure the placement is within bounds
    x1_end = x1_start + w1
    y1_end = y1_start + h1

    # Adjust if the placement goes out of bounds
    if x1_start < 0: 
        x1_start = 0
        x1_end = min(w1, w2)  # Adjust end accordingly
    if y1_start < 0: 
        y1_start = 0
        y1_end = min(h1, h2)  # Adjust end accordingly
    if x1_end > w2: 
        x1_end = w2
        x1_start = max(0, w2 - w1)
    if y1_end > h2: 
        y1_end = h2
        y1_start = max(0, h2 - h1)

    # Overlay foreground onto background
    background[y1_start:y1_end, x1_start:x1_end] = foreground[:y1_end - y1_start, :x1_end - x1_start]

    # Define output file path
    overlayed_binary_carpet_path = os.path.join(temp_path, "overlayed_carpet.jpg")

    # Save the final image
    cv2.imwrite(overlayed_binary_carpet_path, background)
    print(f"014 Image saved as {overlayed_binary_carpet_path}")

    return overlayed_binary_carpet_path

def main():
    room_img_path = "D:/Wrishav/floorOverlay/inputRoom/room4.jpg"
    carpet_img_path = "D:/Wrishav/floorOverlay/carpet/carpet2.jpg"

    overlayed_binary_carpet_path = place_on_black(room_img_path, carpet_img_path)
    print(f"Overlayed Binary Carpet Image Path: {overlayed_binary_carpet_path}")

if __name__ == "__main__":
    main()