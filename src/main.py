# The main business logic and high-level functions
# for the application workflows.

import os
import cv2
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation, AutoImageProcessor, MaskFormerModel
from PIL import Image
from numba import njit, prange

# Internal imports from your new utils module
from utils import order_points, four_point_transform
from utils import scale_room_image, tileDesign
from utils import scale_carpet, place_on_black
from utils import carpet_circle, carpet_ellipse_and_center
from utils import create_wall_overlay, create_floor_overlay

# Global variables for the model
feature_extractor = None
model = None
device = torch.device('cpu')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# ───── From app.py ───────────────────────────────────── #

def overlay_carpet_trapezoid(room_path, carpet_path, output_path="final_out"):
    """
    Overlays a trapezoidal-perspective carpet on the floor of a room.
    """
    # Placeholder for the main logic from the original file
    # This function is now defined in overlay.py and is simply called here.
    return apply_transparency_to_black_background(room_path, carpet_path, overlay_type="trapezoid")

def overlay_carpet_ellipse(room_path, carpet_path, output_path="final_out"):
    """
    Overlays an elliptical-perspective carpet on the floor of a room.
    """
    # Placeholder for the main logic from the original file
    # This function is now defined in overlay.py and is simply called here.
    return apply_transparency_to_black_background(room_path, carpet_path, overlay_type="ellipse")

def apply_transparency_to_black_background(room_image_path, carpet_image_path, overlay_type, carpet_dimensions=None, output_path="../Floor-Overlay/final_out"):
    """
    Combines a room image and a carpet image with a transparent background.
    """
    # This function is now the main entry point for the carpet overlay.
    # The logic from the original overlay.py is now here.
    try:
        room_img = cv2.imread(room_image_path)
        
        if room_img is None:
            raise FileNotFoundError(f"|ERROR| Could not read room image at: {room_image_path}")

        room_image_name = os.path.basename(room_image_path).split('.')[0]
        
        if overlay_type == "trapezoid":
            carpet_img_path = adjust_carpet_perspective(carpet_image_path)
            
        elif overlay_type == "ellipse":
            carpet_img_path, center = carpet_ellipse_and_center(carpet_image_path, room_image_path, center_coords=None)
            
        else:
            scaled_carpet_path = scale_carpet(room_image_path, carpet_image_path, carpet_dimensions)
            carpet_img_path = place_on_black(scaled_carpet_path)
            
        if carpet_img_path is None:
            raise Exception("Failed to get transformed carpet image path.")
        
        overlayed_carpet_img = cv2.imread(carpet_img_path, cv2.IMREAD_UNCHANGED)
        
        if overlayed_carpet_img is None:
            raise FileNotFoundError(f"|ERROR| Could not read overlayed carpet image at: {carpet_img_path}")
        
        alpha_channel = overlayed_carpet_img[:, :, 3] / 255.0
        alpha_channel_3d = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
        
        room_img_float = room_img.astype(np.float32)
        overlayed_carpet_img_float = overlayed_carpet_img.astype(np.float32)
        
        result_float = (overlayed_carpet_img_float * alpha_channel_3d) + \
                       (room_img_float * (1 - alpha_channel_3d))
        
        result = np.clip(result_float, 0, 255).astype(np.uint8)
        
        result_img_path = os.path.join(output_path, f"overlayed_carpet_e_{room_image_name}.jpg")
        cv2.imwrite(result_img_path, result)
        return result_img_path
        
    except Exception as e:
        print(f"|ERROR| An error occurred during transparency application: {str(e)}")
        return None

# ───── From carpet_working.py ───────────────────────────────────── #

def overlay_texture_on_floor(original_image_path, mask_path, tiled_texture_path):
    """
    Overlays a tiled texture onto the floor area of a room image based on a segmentation mask.
    """
    try:
        original_image = cv2.imread(original_image_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        tiled_image = cv2.imread(tiled_texture_path)

        if original_image is None or mask_image is None or tiled_image is None:
            raise FileNotFoundError("|ERROR| One or more input images not found.")

        # Ensure mask is binary (0 or 255)
        _, binary_mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
        
        resized_texture = cv2.resize(tiled_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_AREA)

        # Apply the mask to the texture
        texture_on_floor = cv2.bitwise_and(resized_texture, resized_texture, mask=binary_mask)

        # Invert the mask to get everything BUT the floor
        inverse_mask = cv2.bitwise_not(binary_mask)

        # Apply the inverse mask to the original image
        room_without_floor = cv2.bitwise_and(original_image, original_image, mask=inverse_mask)

        # Combine the two images
        final_result = cv2.add(room_without_floor, texture_on_floor)
        
        return final_result

    except Exception as e:
        print(f"|ERROR| An error occurred during texture overlay: {str(e)}")
        return None

# ───── From convert_binary.py ───────────────────────────────────── #

def convert_to_binary_mask(room_image_path, temp_path="../Floor-Overlay/temporary"):
    """
    Converts a segmentation mask for a room into a binary mask.
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
    
    binary_mask_path = os.path.join(temp_output_dir, f"binary_{os.path.basename(mask_image_path)}")
    
    cv2.imwrite(binary_mask_path, binary_mask)
    print(f"|INFO| Binary mask saved at: {binary_mask_path}")
    
    return binary_mask_path

def convert_to_binary_carpet(carpet_img_path, temp_path="../Floor-Overlay/temporary"):
    """
    Converts a carpet image into a binary mask based on its content.
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

# ───── From find_centroid.py ───────────────────────────────────── #

def find_and_mark_floor_center(room_img_path, temp_path="../Floor-Overlay/temporary"):
    """
    Finds the centroid of the floor mask in a room image and marks it.
    """
    masked_image_path = mask(room_img_path)
    image = cv2.imread(masked_image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            os.makedirs(temp_path, exist_ok=True)
            output_path = os.path.join(temp_path, "marked_masked_image.jpg")
            cv2.imwrite(output_path, image)
            print(f"|INFO| Marked image saved at: {output_path}")
            return (cx, cy)
    print("|WARNING| No floor mask detected.")
    return None

# ───── From floor_mask_model.py ───────────────────────────────────── #

def load_model():
    """
    Loads the MaskFormer model and feature extractor once for the entire application.
    """
    global feature_extractor, model, device
    print("|INFO| Attempting to load MaskFormer model...")
    try:
        feature_extractor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
        model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-coco").to(device)
        print("|INFO| MaskFormer model loaded successfully.")
    except Exception as e:
        print(f"|ERROR| Failed to load MaskFormer model: {e}")
        feature_extractor = None
        model = None
        
def infer(input_image_path, wallitemid, outputpath, temp_path="../Floor-Overlay/temporary"):
    """
    Performs inference using the MaskFormer model to create a segmentation mask.
    """
    global feature_extractor, model, device
    if not all([feature_extractor, model]):
        print("|ERROR| Model not loaded. Cannot perform inference.")
        return 0

    try:
        image = Image.open(input_image_path)
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        result = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_panoptic_map = result.numpy()
    except Exception as e:
        print(f"|ERROR| Inference failed: {e}")
        return 0

    color_predicted_panoptic_map = np.zeros((predicted_panoptic_map.shape[0], predicted_panoptic_map.shape[1], 3), dtype=np.uint8)
    color_predicted_panoptic_map[predicted_panoptic_map == wallitemid ] = (255,0,0)
    plt.imsave(outputpath,color_predicted_panoptic_map)
    print('|INFO| Inference done!')
    if torch.cuda.is_available():
        model.to('cpu')
        del inputs,outputs,result
        torch.cuda.empty_cache()
    print('|INFO| CUDA memory allocated:', torch.cuda.memory_allocated())
    return 1

# ───── From mask_room_image.py ───────────────────────────────────── #

def mask(room_image_path, output_path="../Floor-Overlay/mask_out", temp_path="../Floor-Overlay/temporary"):
    """
    Generates a segmentation mask for the floor of a room image.
    """
    scaled_room_img_path = scale_room_image(room_image_path, temp_path)
    
    if scaled_room_img_path is None:
        return None
    
    unique_id = os.path.basename(scaled_room_img_path).split('.')[0]
    output_mask_path = os.path.join(output_path, f"mask_{unique_id}.jpg")
    
    print(f"|INFO| Generating mask for {scaled_room_img_path}...")
    success = infer(scaled_room_img_path, 0, output_mask_path)
    
    if success:
        print(f"|INFO| Mask generated successfully and saved to {output_mask_path}")
        return output_mask_path
    else:
        print("|ERROR| Failed to generate mask.")
        return None

# ───── From overlay.py ───────────────────────────────────── #

def adjust_carpet_perspective(carpet_img_path, temp_path="../Floor-Overlay/temporary"):
    """
    Applies a perspective transformation to a carpet image to make it look like a trapezoid.
    """
    image = cv2.imread(carpet_img_path)
    h, w = image.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    offset = h // 3
    dst_pts = np.float32([
        [offset, 0],         
        [w - offset, 0],     
        [w, h],              
        [0, h]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h))
    warped_path = os.path.join(temp_path, "warped_carpet.jpg")
    cv2.imwrite(warped_path, warped)
    print(f"|INFO| Perspective-transformed carpet saved to {warped_path}")
    return warped_path