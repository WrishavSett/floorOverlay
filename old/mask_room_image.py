# 011
 
import os
import cv2
import numpy as np
from floor_mask_model import load_model, infer

def scale_room_image(room_image_path,
                     temp_path="../Floor-Overlay/temporary",
                     target_resolution=(1920, 1080)):
    """
    Scales a room image to fit within the target resolution while maintaining aspect ratio.

    Args:
        room_image_path (str): Path to the original room image.
        temp_path (str): Directory to save the scaled image.
        target_resolution (tuple): The desired (width, height) resolution.

    Returns:
        str: Path to the scaled room image.
    """
    image = cv2.imread(room_image_path)
    if image is None:
        raise FileNotFoundError(f"|ERROR| Could not read room image at: {room_image_path}")

    orig_height, orig_width = image.shape[:2]
    target_width, target_height = target_resolution
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height
    scale_factor = min(scale_w, scale_h)

    new_width = max(1, int(orig_width * scale_factor))
    new_height = max(1, int(orig_height * scale_factor))

    if (new_width, new_height) != (orig_width, orig_height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"|INFO| Scaled room image from ({orig_width}, {orig_height}) to ({new_width}, {new_height})")
    else:
        print(f"|INFO| Room image already at target resolution ({orig_width}, {orig_height})")

    os.makedirs(temp_path, exist_ok=True)
    scaled_image_path = os.path.join(temp_path, "scaled_room_image.jpg")
    cv2.imwrite(scaled_image_path, image)
    print(f"|INFO| Scaled room image saved at: {scaled_image_path}")

    return scaled_image_path

def mask(room_image_path):
    """
    Generates a segmentation mask for the floor of a room image.

    Args:
        room_image_path (str): Path to the room image.

    Returns:
        str or None: Path to the generated mask image if successful, None otherwise.
    """
    room_image_name = os.path.splitext(os.path.basename(room_image_path))[0]
    mask_output_dir = "../Floor-Overlay/mask_out"
    os.makedirs(mask_output_dir, exist_ok=True)
    mask_output_path = os.path.join(mask_output_dir, f"{room_image_name}_mask.jpg")

    # load_model()
    success = infer(room_image_path, 0, mask_output_path)
    
    if success:
        print("|INFO| Inference completed successfully. Proceeding with texture application...")
        return mask_output_path
    else:
        print("|WARNING| Feature not found in image. Exiting...")
        return None

def tileDesign(design_path,
               multiplier=5,
               temp_path="../Floor-Overlay/temporary"):
    """
    Tiles a design image and saves the result to a temporary location.

    Args:
        design_path (str): Path to the design image.
        multiplier (int): The number of times to repeat the tile.
        temp_path (str): The directory to save the temporary tiled image.

    Returns:
        str or None: Path to the saved tiled image or None if the image cannot be read.
    """
    print(f"|INFO| Tiling design from {design_path} with a multiplier of {multiplier}...")

    design_img = cv2.imread(design_path)
    if design_img is None:
        print(f"|ERROR| Could not read image at path: {design_path}")
        return None

    tile_height, tile_width, _ = design_img.shape
    target_width = tile_width * multiplier
    target_height = tile_height * multiplier

    print(f"|INFO| Tiling {multiplier}x horizontally and {multiplier}x vertically.")

    tiled_image = np.tile(design_img, (multiplier, multiplier, 1))
    tiled_image_cropped = tiled_image[0:target_height, 0:target_width]

    os.makedirs(temp_path, exist_ok=True)
    tiled_image_path = os.path.join(temp_path, "tiled_design.jpg")
    cv2.imwrite(tiled_image_path, tiled_image_cropped)

    print(f"|INFO| Design successfully tiled and saved to {tiled_image_path}.")
    return tiled_image_path

def main():
    """
    Main function to orchestrate the image processing workflow.
    """
    room_image_path = "../Floor-Overlay/inputRoom/room4.jpg"
    design_image_path = "../Floor-Overlay/sample_images/designs/tile10.jpg"

    # Step 1: Generate a mask for the room image
    mask_output_path = mask(room_image_path)
    
    # Step 2: Tile the design image
    tiled_design_path = tileDesign(design_image_path, multiplier=4)

    if mask_output_path and tiled_design_path:
        print(f"|OUTPUT| Mask path: {mask_output_path}")
        print(f"|OUTPUT| Tiled design path: {tiled_design_path}")
        # Note: The original main function was missing the step to
        # combine the room image, mask, and tiled design.
        # This clean-up assumes those steps would be added here.
    else:
        print("|ERROR| Workflow failed due to a missing mask or tiled design.")

if __name__ == "__main__":
    main()