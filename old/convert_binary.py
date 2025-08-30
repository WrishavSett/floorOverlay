# 012

import os
import cv2
from mask_room_image import mask

def convert_to_binary_mask(room_image_path, temp_path="../Floor-Overlay/temporary"):
    """
    Converts a segmentation mask for a room into a binary mask.

    Args:
        room_image_path (str): The path to the original room image.
        temp_path (str): The temporary path to store the binary mask.

    Returns:
        str or None: The path to the saved binary mask, or None if the process fails.
    """
    mask_image_path = mask(room_image_path)
    
    if not mask_image_path:
        print("|ERROR| Masking failed. Exiting...")
        return None
    
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        print("|ERROR| Failed to read mask image. Exiting...")
        return None
    
    blurred_mask = cv2.GaussianBlur(mask_image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred_mask, 1, 255, cv2.THRESH_BINARY)
    
    temp_output_dir = temp_path
    os.makedirs(temp_output_dir, exist_ok=True)
    
    binary_mask_path = os.path.join(temp_output_dir, f"room_binary_mask.jpg")
    
    cv2.imwrite(binary_mask_path, binary_mask)
    print(f"|INFO| Binary mask saved at: {binary_mask_path}")
    
    return binary_mask_path

def convert_to_binary_carpet(carpet_img_path, temp_path="../Floor-Overlay/temporary"):
    """
    Converts a carpet image to a binary mask.

    Args:
        carpet_img_path (str): The path to the original carpet image.
        temp_path (str): The temporary path to store the binary mask.

    Returns:
        str or None: The path to the saved binary mask, or None if the process fails.
    """
    carpet_image = cv2.imread(carpet_img_path, cv2.IMREAD_GRAYSCALE)
    if carpet_image is None:
        print("|ERROR| Failed to read carpet image. Exiting...")
        return None
    
    blurred_carpet = cv2.GaussianBlur(carpet_image, (5, 5), 0)
    _, binary_carpet = cv2.threshold(blurred_carpet, 1, 255, cv2.THRESH_BINARY)

    temp_output_dir = temp_path
    os.makedirs(temp_output_dir, exist_ok=True)
    
    binary_carpet_path = os.path.join(temp_output_dir, f"carpet_binary_mask.jpg")
    
    cv2.imwrite(binary_carpet_path, binary_carpet)
    print(f"|INFO| Binary carpet mask saved at: {binary_carpet_path}")
    
    return binary_carpet_path

def main():
    """
    Main function to demonstrate the binary mask conversion process for a room and a carpet.
    """
    room_bin_mask_path = convert_to_binary_mask("../Floor-Overlay/inputRoom/room4.jpg")
    if room_bin_mask_path:
        print(f"|OUTPUT| Room binary mask created at: {room_bin_mask_path}")
    else:
        print("|ERROR| Failed to create the room binary mask.")

    carpet_bin_mask_path = convert_to_binary_carpet("../Floor-Overlay/carpet/carpet1.jpg")
    if carpet_bin_mask_path:
        print(f"|OUTPUT| Carpet binary mask created at: {carpet_bin_mask_path}")
    else:
        print("|ERROR| Failed to create the carpet binary mask.")

if __name__ == "__main__":
    main()