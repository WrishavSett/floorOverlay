# 011
 
import os
import cv2
from floor_mask_model import load_model, infer

def mask(room_image_path):
    # Example paths for the images and output
    room_image_path = room_image_path

    # Extract the room image name without extension
    room_image_name = os.path.splitext(os.path.basename(room_image_path))[0]
    
    # Define mask output path with new format
    mask_output_dir = "../floorOverlay/mask_out"
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
    path = mask("../floorOverlay/inputRoom/room4.jpg")
    print(path)

if __name__ == "__main__":
    main()