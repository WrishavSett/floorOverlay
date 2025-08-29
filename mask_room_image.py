# 011
 
import os
import cv2
import numpy as np
from floor_mask_model import load_model, infer

def scale_room_image(room_image_path,
                     temp_path="../Floor-Overlay/temporary",
                     target_resolution=(1920, 1080)):
    """
    Scales a room image to fit within the target resolution (1920x1080)
    while maintaining aspect ratio. Upscales if below target, downscales if above.

    Args:
        room_image_path (str): Path to the original room image.
        temp_path (str): Directory to save the scaled image.
        target_resolution (tuple): Desired (width, height) resolution.

    Returns:
        str: Path to the scaled room image saved in the temporary folder.
    """
    # Load image
    image = cv2.imread(room_image_path)
    if image is None:
        raise FileNotFoundError(f"011 Could not read room image at: {room_image_path}")

    orig_height, orig_width = image.shape[:2]
    target_width, target_height = target_resolution

    # Compute scale factor to fit image within target resolution
    scale_w = target_width / orig_width
    scale_h = target_height / orig_height
    scale_factor = min(scale_w, scale_h)  # fit both width and height

    # Calculate new dimensions
    new_width = max(1, int(orig_width * scale_factor))
    new_height = max(1, int(orig_height * scale_factor))

    # Resize image
    if (new_width, new_height) != (orig_width, orig_height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"011 Scaled room image from ({orig_width}, {orig_height}) to ({new_width}, {new_height})")
    else:
        print(f"011 Room image already at target resolution ({orig_width}, {orig_height})")

    # Ensure output folder exists
    os.makedirs(temp_path, exist_ok=True)
    scaled_image_path = os.path.join(temp_path, "scaled_room_image.jpg")
    cv2.imwrite(scaled_image_path, image)
    print(f"011 Scaled room image saved at: {scaled_image_path}")

    return scaled_image_path

def mask(room_image_path):
    # Example paths for the images and output
    room_image_path = room_image_path

    # Extract the room image name without extension
    room_image_name = os.path.splitext(os.path.basename(room_image_path))[0]
    
    # Define mask output path with new format
    mask_output_dir = "../Floor-Overlay/mask_out"
    os.makedirs(mask_output_dir, exist_ok=True)
    mask_output_path = os.path.join(mask_output_dir, f"{room_image_name}_mask.jpg")

    # Load the model
    # load_model()

    # Perform inference
    success = infer(room_image_path, 0, mask_output_path)
    
    if success:
        print("011 Inference completed successfully. Proceeding with texture application...")
        return mask_output_path
    else:
        print("011 Feature not found in image. Exiting...")
        return None

def tileDesign(design_path,
               multiplier=5,
               temp_path="../Floor-Overlay/temporary"):
    """
    Tiles a design or texture image by a specific multiplier using OpenCV,
    saves it to a temporary location, and returns the path.

    Args:
        design_path (str): The file path to the design/tile image.
        multiplier (int): The number of times to repeat the tile horizontally and vertically.
        temp_path (str): The directory to save the temporary tiled image.

    Returns:
        str: The file path to the saved tiled image.
             Returns None if the design image cannot be read.
    """
    print(f"017 Tiling design from {design_path} with a multiplier of {multiplier}...")

    # Read the design image
    design_img = cv2.imread(design_path)
    if design_img is None:
        print(f"017 Error: Could not read image at path: {design_path}")
        return None

    # Get the dimensions of the single tile
    tile_height, tile_width, _ = design_img.shape

    # Calculate the new target dimensions based on the multiplier
    target_width = tile_width * multiplier
    target_height = tile_height * multiplier

    print(f"017 Tiling {multiplier}x horizontally and {multiplier}x vertically.")

    # Create the tiled image
    tiled_image = np.tile(design_img, (multiplier, multiplier, 1))

    # Crop the tiled image to the target dimensions (which are now based on the multiplier)
    tiled_image_cropped = tiled_image[0:target_height, 0:target_width]

    # Ensure the output directory exists
    os.makedirs(temp_path, exist_ok=True)
    
    # Save the tiled image to a temporary file
    tiled_image_path = os.path.join(temp_path, "tiled_design.jpg")
    cv2.imwrite(tiled_image_path, tiled_image_cropped)

    print(f"017 Design successfully tiled and saved to {tiled_image_path}.")
    return tiled_image_path

def main():
    # path = mask("../Floor-Overlay/inputRoom/room4.jpg")
    # print(path)

    sample_design_path = "../Floor-Overlay/sample_images/designs/tile10.jpg"
    
    # Use a multiplier instead of target dimensions
    tile_multiplier = 4
    tiled_result_path = tileDesign(sample_design_path)

    if tiled_result_path is not None:
        print(f"017 Returned path: {tiled_result_path}")
        # Load the saved image to verify
        tiled_image = cv2.imread(tiled_result_path)
        cv2.imshow("Tiled Design", tiled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("017 Tiled image displayed. Press any key to close.")

if __name__ == "__main__":
    main()