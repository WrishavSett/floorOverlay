# 012

import os
import cv2
from mask_room_image import mask

def convert_to_binary_mask(room_image_path, temp_path="../floorOverlay/temporary"):
    # Get the mask image path
    mask_image_path = mask(room_image_path)
    
    if not mask_image_path:
        print("012 Masking failed. Exiting...")
        return None
    
    # Read the mask image
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        print("012 Failed to read mask image. Exiting...")
        return None
    
    # Convert the mask to binary (thresholding)
    blurred_mask = cv2.GaussianBlur(mask_image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred_mask, 1, 255, cv2.THRESH_BINARY)
    
    # Define output directory and filename
    temp_output_dir = temp_path
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # mask_image_name = os.path.splitext(os.path.basename(mask_image_path))[0] # Redundant, Generalised NAME
    binary_mask_path = os.path.join(temp_output_dir, f"room_binary_mask.jpg")
    
    # Save the binary mask image
    cv2.imwrite(binary_mask_path, binary_mask)
    print(f"012 Binary mask saved at: {binary_mask_path}")
    
    return binary_mask_path

def convert_to_binary_carpet(carpet_img_path, temp_path="../floorOverlay/temporary"):
    # Read the carpet image
    carpet_image = cv2.imread(carpet_img_path, cv2.IMREAD_GRAYSCALE)
    if carpet_image is None:
        print("012 Failed to read carpet image. Exiting...")
        return None
    
    # Convert the carpet image to binary (thresholding)
    blurred_carpet = cv2.GaussianBlur(carpet_image, (5, 5), 0)
    _, binary_carpet = cv2.threshold(blurred_carpet, 1, 255, cv2.THRESH_BINARY)

    # Define output directory and filename
    temp_output_dir = temp_path
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # carpet_image_name = os.path.splitext(os.path.basename(carpet_img_path))[0]
    binary_carpet_path = os.path.join(temp_output_dir, f"carpet_binary_mask.jpg")
    
    # Save the binary carpet image
    cv2.imwrite(binary_carpet_path, binary_carpet)
    print(f"012 Binary carpet mask saved at: {binary_carpet_path}")
    
    return binary_carpet_path

def main():
    room_bin_mask_path = convert_to_binary_mask("../floorOverlay/inputRoom/room4.jpg", temp_path="../floorOverlay/temporary")
    print(room_bin_mask_path)
    
    carpet_original_bin_mask_path = convert_to_binary_carpet("../floorOverlay/carpet/carpet1.jpg", temp_path="../floorOverlay/temporary")
    print(carpet_original_bin_mask_path)


if __name__ == "__main__":
    main()