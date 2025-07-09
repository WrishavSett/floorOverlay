# 003

import os
import cv2
from floor_mask_model import load_model, infer
from carpet_working import overlay_texture_on_floor

def main():
    # Example paths for the images and output
    room_image_path = "../floorOverlay/inputRoom/room1.jpg"
    design_image_path = "../floorOverlay/inputTile/tile1.jpg"
    
    # Extract the room image name without extension
    room_image_name = os.path.splitext(os.path.basename(room_image_path))[0]
    design_image_name = os.path.splitext(os.path.basename(design_image_path))[0]
    
    # Define mask output path with new format
    mask_output_dir = "../floorOverlay/mask_out"
    os.makedirs(mask_output_dir, exist_ok=True)
    mask_output_path = os.path.join(mask_output_dir, f"{room_image_name}_mask.jpg")

    # Define the output directory for final results
    final_output_dir = "../floorOverlay/final_out"
    os.makedirs(final_output_dir, exist_ok=True)
    final_output_path = os.path.join(final_output_dir, f"{room_image_name}_{design_image_name}_output.jpg")

    # Load the model
    load_model()

    # Perform inference
    success = infer(room_image_path, 0, mask_output_path)
    
    if success:
        print("Inference completed successfully. Proceeding with texture application...")
        
        # Apply texture overlay
        final_output = overlay_texture_on_floor(room_image_path, mask_output_path, design_image_path)
        
        if final_output is not None:
            # Save the final output
            cv2.imwrite(final_output_path, final_output)
            print(f"Final output saved at: {final_output_path}")
            
            # Display the result
            cv2.imshow("Scene with Perspective-Warped Texture on Floor", final_output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to generate final output.")
    else:
        print("Feature not found in image. Exiting...")

if __name__ == "__main__":
    main()
