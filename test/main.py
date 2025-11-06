# main.py

import os
import cv2
import torch
import requests
import numpy as np
from PIL import Image
from numba import njit, prange
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, MaskFormerModel
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation

from utils import *

model = None
feature_extractor = None
device = torch.device('cpu')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

@njit(parallel=True)
def create_wall_overlay(mask, dsgn,woverlay):
    """
    Overlays a design pattern onto a mask, creating a new image with the design.

    Args:
        mask (np.array): The segmentation mask.
        dsgn (np.array): The design pattern image.
        woverlay (np.array): The output image to be created.

    Returns:
        np.array: The image with the design overlaid.
    """
    w,h,_ = woverlay.shape
    dw,dh,_ = dsgn.shape
    for i in prange(0,w):
        for j in prange(0,h):
            if(mask[i][j][0] == 255):
                p = dsgn[i%dw][j%dh]
                woverlay[i][j]= p
    return woverlay

@njit(parallel=True)
def create_output_image(imagearray, walloverlayarray):
    """
    Merges a wall overlay onto the original image.

    Args:
        imagearray (np.array): The original image array.
        walloverlayarray (np.array): The overlay image array.

    Returns:
        np.array: The combined image with the overlay.
    """
    h,w,_ =  walloverlayarray.shape
    for i in prange(0,h):
        for j in prange(0,w):
            if(walloverlayarray[i][j].sum() > 0 ):
                imagearray[i][j] =  walloverlayarray[i][j]
    return imagearray.astype(np.uint8)

@njit(parallel=True)
def create_image_with_shadow(img_gray, hsv_image, walloverlayarray):
    """
    Adds a shadow effect to the wall overlay based on the original image's grayscale values.

    Args:
        img_gray (np.array): The grayscale version of the original image.
        hsv_image (np.array): The image converted to HSV color space.
        walloverlayarray (np.array): The wall overlay image.

    Returns:
        np.array: The HSV image with shadows applied.
    """
    h,w,_ =  hsv_image.shape
    hsvmin = np.min(hsv_image[:,:,2])
    hsvmax = np.max(hsv_image[:,:,2])
    for i in prange(0,h):
        for j in prange(0,w):
            if(walloverlayarray[i][j].sum() > 0 ):
                hsv_image[i][j][2] = abs(hsv_image[i][j][2] - (((img_gray[i][j]/1)-hsvmin)/(hsvmax-hsvmin))*100)
    print('|INFO| The shape of the HSV image is:', hsv_image.shape)
    return hsv_image.astype(np.uint8)

def load_model():
    """
    Initializes and loads the MaskFormer model and feature extractor.

    Returns:
        None
    """
    global feature_extractor,model,device
    # load MaskFormer fine-tuned on COCO panoptic segmentation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
    # model.to(device)
    print("|INFO| Model Successfully Loaded")
    # image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
    # model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")

def infer(imagepath, designimgpath, outputpath, mode=3):
    """
    Performs inference on an image to generate a segmentation mask for a specified feature.

    Args:
        imagepath (str): The path to the input image.
        designimgpath (str): The path to the design image (unused in this function).
        outputpath (str): The path to save the output mask image.
        mode (int, optional): The label ID for the feature to segment. Defaults to 3 (floors).

    Returns:
        int: 1 if inference is successful, 0 otherwise.
    """
    #mode 0 for walls
    #model 3 for floors
    #model 28 for carpet
    global feature_extractor,model,device
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(device)
    image = Image.open(imagepath).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    # inputs = feature_extractor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    # class_queries_logits = outputs.class_queries_logits
    # masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to feature_extractor for postprocessing
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_panoptic_map = result["segmentation"].cpu()

    # Checking if the requested feature is in the image 
    if (mode not in [info['label_id'] for info in result['segments_info']]):
        print(f"|WARNING| The requested feature with mode {mode} was not found in the image.")
        return 0

    # Finding the id of the wall from the segment predictions
    # facebook/maskformer-swin-base-coco" -> 131
    # facebook/maskformer-swin-base-ade => 0
    wallitem = next(item for item in result['segments_info'] if item["label_id"] == mode)
    wallitemid = wallitem['id']

    #creating empty panoptic map
    color_predicted_panoptic_map = np.zeros((predicted_panoptic_map.shape[0], predicted_panoptic_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    color_predicted_panoptic_map[predicted_panoptic_map == wallitemid ] = (255,0,0)

    plt.imsave(outputpath,color_predicted_panoptic_map)
    print('|INFO| Inference done!')
    if torch.cuda.is_available():
        model.to('cpu')
        del inputs,outputs,result
        torch.cuda.empty_cache()
    print('|INFO| CUDA memory allocated:', torch.cuda.memory_allocated())
    return 1

def order_points(pts):
    """
    Orders the corner points of a quadrilateral in a consistent manner (top-left, top-right, bottom-right, bottom-left).

    Args:
        pts (np.array): A NumPy array of shape (N, 2) representing N points.

    Returns:
        np.array: A NumPy array of shape (4, 2) containing the ordered corner points.
    """
    rect = np.zeros((4, 2), dtype="int32")
    points = np.array(pts)
    sorted_points = points[np.argsort(points[:, 1])]
    y_coords = sorted_points[:, 1]
    y_diffs = np.diff(y_coords)
    threshold_index = np.argmax(y_diffs)
    split_value = y_coords[threshold_index]
    top_points = sorted_points[sorted_points[:, 1] <= split_value]
    bottom_points = sorted_points[sorted_points[:, 1] > split_value]
    min_y_index = np.argmin(top_points[:, 1])
    lowest_y_point = top_points[min_y_index]
    top_left = min(top_points, key=lambda p: p[0])
    top_right = max(top_points, key=lambda p: p[0])
    top_left = np.array(top_left)
    top_right = np.array(top_right)
    top_left[1] = lowest_y_point[1]
    top_right[1] = lowest_y_point[1]
    max_y_index = np.argmax(bottom_points[:, 1])
    highest_y_point = bottom_points[max_y_index]
    bottom_left = min(bottom_points, key=lambda p: p[0])
    bottom_right = max(bottom_points, key=lambda p: p[0])
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)
    bottom_left[1] = highest_y_point[1]
    bottom_right[1] = highest_y_point[1]
    rect[0], rect[1], rect[2], rect[3] = top_left, top_right, bottom_right, bottom_left
    return rect

def find_floor_contour(mask_path):
    """
    Finds the largest contour representing the floor in a given binary mask image.

    Args:
        mask_path (str): The file path to the binary mask image.

    Returns:
        tuple: A tuple containing the approximated contour points (np.array) and the binary mask (np.array), or None if no contours are found.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("|ERROR| No contours found in the provided mask image.")
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    approx = cv2.convexHull(largest_contour)
    return approx.reshape(-1, 2), binary_mask

def apply_homography(tile_img, ordered_corners, mask_shape):
    """
    Applies homography to warp a tile image onto a quadrilateral defined by ordered corners.

    Args:
        tile_img (np.array): The source tile image.
        ordered_corners (np.array): An array of the four ordered corner points of the destination area.
        mask_shape (tuple): The shape (height, width) of the target mask or image.

    Returns:
        np.array: The warped image.
    """
    tile_h, tile_w = tile_img.shape[:2]
    src_pts = np.array([[0, 0], [tile_w, 0], [tile_w, tile_h], [0, tile_h]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, ordered_corners)
    return cv2.warpPerspective(tile_img, H, (mask_shape[1], mask_shape[0]))

def overlay_texture_on_floor(original_image_path, mask_path, tile_path):
    """
    Overlays a tile texture onto the floor area detected in a mask.

    Args:
        original_image_path (str): The path to the original image.
        mask_path (str): The path to the segmentation mask of the floor.
        tile_path (str): The path to the tile texture image.

    Returns:
        np.array: The final image with the floor texture overlaid, or None if the process fails.
    """
    corners, binary_mask = find_floor_contour(mask_path)
    if corners is None:
        return None
    ordered_corners = order_points(corners)
    original_image = cv2.imread(original_image_path)
    tile = cv2.imread(tile_path)
    if original_image is None or tile is None:
        print("|ERROR| Could not read one of the input images. Please check the paths.")
        return None
    
    tiled_image = np.tile(tile, (2, 2, 1))
    warped_tile = apply_homography(tiled_image, ordered_corners, binary_mask.shape)
    
    carpet_mask = cv2.bitwise_not(cv2.cvtColor(warped_tile, cv2.COLOR_BGR2GRAY))
    uncovered_mask = cv2.bitwise_and(binary_mask, cv2.threshold(carpet_mask, 250, 255, cv2.THRESH_BINARY)[1])
    resized_mask = cv2.resize(tiled_image, (uncovered_mask.shape[1], uncovered_mask.shape[0]))
    tresult = np.where(uncovered_mask[:, :, None] == 255, resized_mask, warped_tile)
    final_result = np.where(binary_mask[:, :, None] == 255, tresult, original_image)
    return final_result

def overlay_carpet_on_room(original_image_path, carpet_image_path, floor_dimensions, carpet_dimensions, mask_path):
    """
    Overlays a carpet onto a room image with perspective correction based on provided dimensions.

    Args:
        original_image_path (str): The path to the original room image.
        carpet_image_path (str): The path to the carpet image.
        floor_dimensions (str): The dimensions of the floor in "width/height" format (e.g., "10/12").
        carpet_dimensions (str): The dimensions of the carpet in "width/height" format (e.g., "8/10").
        mask_path (str): The path to the floor segmentation mask.

    Returns:
        np.array: The final image with the carpet overlaid, or None if the process fails.
    """
    # 1. Find floor contour
    corners, binary_mask = find_floor_contour(mask_path)
    if corners is None:
        print("|ERROR| Could not find floor contour in the mask.")
        return None
    ordered_corners = order_points(corners)

    # 2. Define the "unwarped" floor rectangle from user dimensions
    try:
        floor_w_ft, floor_h_ft = map(float, floor_dimensions.split('/'))
        carpet_w_ft, carpet_h_ft = map(float, carpet_dimensions.split('/'))
    except ValueError:
        print(f"|ERROR| Invalid format for dimensions. Use 'width/height'.")
        return None

    unwarped_floor_pts = np.array([
        [0, 0], [floor_w_ft, 0], [floor_w_ft, floor_h_ft], [0, floor_h_ft]
    ], dtype=np.float32)

    # 3. Find homography from unwarped floor to warped floor in image
    H, _ = cv2.findHomography(unwarped_floor_pts, ordered_corners)
    if H is None:
        print("|ERROR| Could not compute homography. Check contour points.")
        return None

    # 4. Define the carpet rectangle in the "unwarped" space (centered)
    carpet_x_start = (floor_w_ft - carpet_w_ft) / 2
    carpet_y_start = (floor_h_ft - carpet_h_ft) / 2

    unwarped_carpet_pts = np.array([
        [carpet_x_start, carpet_y_start],
        [carpet_x_start + carpet_w_ft, carpet_y_start],
        [carpet_x_start + carpet_w_ft, carpet_y_start + carpet_h_ft],
        [carpet_x_start, carpet_y_start + carpet_h_ft]
    ], dtype=np.float32)

    # 5. Transform carpet corners to image space
    warped_carpet_pts = cv2.perspectiveTransform(unwarped_carpet_pts.reshape(-1, 1, 2), H)

    # 6. Warp the carpet image to the transformed quadrilateral
    carpet_img = cv2.imread(carpet_image_path)
    if carpet_img is None:
        print(f"|ERROR| Could not read carpet image at: {carpet_image_path}")
        return None
        
    carpet_h, carpet_w = carpet_img.shape[:2]
    original_carpet_pts = np.array([
        [0, 0], [carpet_w, 0], [carpet_w, carpet_h], [0, carpet_h]
    ], dtype=np.float32)

    carpet_warp_matrix, _ = cv2.findHomography(original_carpet_pts, warped_carpet_pts)
    if carpet_warp_matrix is None:
        print("|ERROR| Could not compute carpet warp matrix.")
        return None

    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"|ERROR| Could not read original image at: {original_image_path}")
        return None

    h, w = original_image.shape[:2]
    warped_carpet = cv2.warpPerspective(carpet_img, carpet_warp_matrix, (w, h))

    # Convert to BGRA
    warped_carpet_bgra = cv2.cvtColor(warped_carpet, cv2.COLOR_BGR2BGRA)

    # Create an empty alpha channel mask
    alpha_channel = np.zeros(warped_carpet.shape[:2], dtype=np.uint8)

    # Fill the polygon defined by warped_carpet_pts with white (255) to create the carpet's mask
    cv2.fillConvexPoly(alpha_channel, np.int32(warped_carpet_pts), 255)

    # Apply this alpha channel to the BGRA image
    warped_carpet_bgra[:, :, 3] = alpha_channel

    return warped_carpet_bgra