# 011
 
import os
import cv2
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

def main():
    path = mask("../Floor-Overlay/inputRoom/room4.jpg")
    print(path)

if __name__ == "__main__":
    main()