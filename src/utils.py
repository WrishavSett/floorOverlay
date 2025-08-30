# All general-purpose helper and utility functions
# that can be used by various modules.

import os
import cv2
import base64
import numpy as np
import pandas as pd
from PIL import Image
from numba import njit, prange
import requests
from io import BytesIO

# ───── From app.py ───────────────────────────────────── #

def decode_base64_to_image(base64_string):
    """
    Decodes a base64 string into an OpenCV image (numpy array).
    """
    try:
        if "data:image" in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"|ERROR| Failed to decode base64 string: {e}")
        return None

def encode_image_to_base64(image):
    """
    Encodes an OpenCV image (numpy array) to a base64 string.
    """
    if image is None:
        return ""
    
    _, buffer = cv2.imencode('.png', image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return base64_string

# ───── From test_app.py ───────────────────────────────────── #

def image_to_base64(image_path):
    """
    Encodes an image file to a base64 string.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def save_base64_image(base64_string, filename):
    """
    Saves a base64-encoded string as an image file.
    """
    img_data = base64.b64decode(base64_string)
    with open(filename, "wb") as img_file:
        img_file.write(img_data)

# ───── From carpet_circle.py ───────────────────────────────────── #

def carpet_circle(carpet_img_path, temp_path="../Floor-Overlay/temporary"):
    """
    Crops a carpet image into a circle.
    """
    carpet_img = cv2.imread(carpet_img_path)
    
    if carpet_img is None:
        raise FileNotFoundError(f"Could not read image at path: {carpet_img_path}")
    
    height, width = carpet_img.shape[:2]

    if carpet_img.shape[2] == 3:
        carpet_img = cv2.cvtColor(carpet_img, cv2.COLOR_BGR2BGRA)

    center = (width // 2, height // 2)
    radius = min(width, height) // 2

    mask = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.circle(mask, center, radius, (255, 255, 255, 255), -1)

    result = cv2.bitwise_and(carpet_img, mask)

    output_name = f"circular_{os.path.basename(carpet_img_path)}"
    circular_carpet_path = os.path.join(temp_path, output_name)
    cv2.imwrite(circular_carpet_path, result)
    
    print(f"|INFO| Circular carpet saved to {circular_carpet_path}")

    return circular_carpet_path

def carpet_ellipse_and_center(carpet_img_path, room_img_path, temp_path="../Floor-Overlay/temporary", center_coords=None):
    """
    Applies a perspective transformation to an image to make it look like an ellipse in a room.
    
    Args:
        carpet_img_path (str): The path to the carpet image.
        room_img_path (str): The path to the room image, used for scaling.
        temp_path (str): The path to the temporary directory.
        center_coords (tuple, optional): A tuple (x, y) for the ellipse center. If None, the center is calculated.

    Returns:
        tuple: A tuple containing the path to the saved elliptical carpet image and the center coordinates (x, y).
    """
    # Use the scale_carpet utility function
    scaled_carpet_path = scale_carpet(room_img_path, carpet_img_path)
    scaled_carpet = cv2.imread(scaled_carpet_path)
    
    if scaled_carpet is None:
        print(f"|ERROR| Could not read scaled carpet image at: {scaled_carpet_path}")
        return None, None
        
    height, width, _ = scaled_carpet.shape
    
    # Define source and destination points for a standard perspective transform to create an ellipse
    src_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    # Destination points for a stretched ellipse
    dst_pts = np.float32([
        [width * 0.1, height * 0.1], 
        [width * 0.9, height * 0.1], 
        [width, height], 
        [0, height]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(scaled_carpet, M, (width, height))

    if center_coords is not None:
        center = center_coords
    else:
        # Calculate the center of the warped image
        center_x = int((dst_pts[0][0] + dst_pts[1][0] + dst_pts[2][0] + dst_pts[3][0]) / 4)
        center_y = int((dst_pts[0][1] + dst_pts[1][1] + dst_pts[2][1] + dst_pts[3][1]) / 4)
        center = (center_x, center_y)

    print(f"|INFO| Center of the ellipse: {center}")

    if warped.shape[2] == 4:
        alpha_channel = warped[:, :, 3]
        rgb_channels = warped[:, :, :3]
        mask = alpha_channel == 0
        rgb_channels[mask] = [0, 0, 0]
        warped = rgb_channels

    output_name = "carpet_ellipse.jpg"
    carpet_ellipse_path = os.path.join(temp_path, output_name)
    cv2.imwrite(carpet_ellipse_path, warped)
    print(f"|INFO| Horizontally-stretched 3D perspective carpet saved to {carpet_ellipse_path}")

    return carpet_ellipse_path, center

# ───── From carpet_working.py ───────────────────────────────────── #

def order_points(pts):
    """
    Orders the corner points of a quadrilateral in a consistent manner (top-left, top-right, bottom-right, bottom-left).
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
    bottom_left = min(bottom_points, key=lambda p: p[0])
    bottom_right = max(bottom_points, key=lambda p: p[0])
    bottom_left = np.array(bottom_left)
    bottom_right = np.array(bottom_right)
    rect[0] = top_left
    rect[1] = top_right
    rect[2] = bottom_right
    rect[3] = bottom_left
    return rect

def four_point_transform(image, pts):
    """
    Applies a four-point perspective transformation to an image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# ───── From floor_mask_model.py ───────────────────────────────────── #

@njit(parallel=True)
def create_wall_overlay(mask,dsgn,woverlay):
    """
    Overlays a design pattern onto a mask, creating a new image with the design.
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
def create_floor_overlay(mask, dsgn, foverlay):
    """
    Overlays a design pattern onto a floor mask.
    """
    w,h,_ = foverlay.shape
    dw,dh,_ = dsgn.shape
    for i in prange(0,w):
        for j in prange(0,h):
            if(mask[i][j][0]==255):
                p = dsgn[i%dw][j%dh]
                foverlay[i][j]=p
    return foverlay

# ───── From mask_room_image.py ───────────────────────────────────── #

def scale_room_image(room_image_path,
                     temp_path="../Floor-Overlay/temporary",
                     target_resolution=(1920, 1080)):
    """
    Scales a room image to fit within the target resolution while maintaining aspect ratio.
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
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        scaled_image_path = os.path.join(temp_path, f"scaled_{os.path.basename(room_image_path)}")
        cv2.imwrite(scaled_image_path, scaled_image)
        
        print(f"|INFO| Scaled image saved to {scaled_image_path}")
        return scaled_image_path
    else:
        print(f"|INFO| Image already at target resolution, no scaling needed.")
        return room_image_path

def tileDesign(design_img_path, multiplier=4, temp_path="../Floor-Overlay/temporary"):
    """
    Tiles a design image multiple times to create a larger pattern.
    """
    design_img = cv2.imread(design_img_path)
    if design_img is None:
        raise FileNotFoundError(f"|ERROR| Could not read design image at: {design_img_path}")
    
    tile_height, tile_width = design_img.shape[:2]
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

# ───── From scale_and_overlay.py ───────────────────────────────────── #

def scale_carpet(room_img_path, carpet_img_path, carpet_dimensions=None, temp_path="../Floor-Overlay/temporary"):
    """
    Scales a carpet image relative to a room image based on provided dimensions or a default ratio.
    """
    ref_image = cv2.imread(room_img_path)
    ref_height, ref_width = ref_image.shape[:2]

    if carpet_dimensions:
        try:
            width_ft, height_ft = map(float, carpet_dimensions.split("/"))
            print(f"|INFO| Scaling carpet with dimensions: {width_ft}ft x {height_ft}ft")
            aspect_ratio = width_ft / height_ft
            max_height = max(1, ref_height // 3)
            max_width = int(max_height * aspect_ratio)

        except ValueError:
            print("|WARNING| Invalid carpet dimensions format. Using default scaling.")
            max_height = ref_height // 3
            max_width = ref_width // 3
    else:
        print("|INFO| No carpet dimensions provided. Using default scaling.")
        max_height = ref_height // 3
        max_width = ref_width // 3

    carpet_image = cv2.imread(carpet_img_path, cv2.IMREAD_UNCHANGED)
    if carpet_image is None:
        raise FileNotFoundError(f"|ERROR| Could not read carpet image at: {carpet_img_path}")

    h, w = carpet_image.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale_factor = min(scale_w, scale_h)

    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)

    scaled_carpet = cv2.resize(carpet_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    output_name = f"scaled_{os.path.basename(carpet_img_path)}"
    scaled_carpet_path = os.path.join(temp_path, output_name)
    cv2.imwrite(scaled_carpet_path, scaled_carpet)
    print(f"|INFO| Scaled carpet saved to {scaled_carpet_path}")

    return scaled_carpet_path

def place_on_black(carpet_img_path, temp_path="../Floor-Overlay/temporary"):
    """
    Places a carpet image with a transparent background onto a solid black background.
    """
    image = cv2.imread(carpet_img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"|ERROR| Could not read image at path: {carpet_img_path}")
    
    if image.shape[2] == 4:
        # Create a solid black background
        black_background = np.zeros_like(image)
        # Separate the alpha channel
        alpha = image[:, :, 3]
        # Copy the BGR channels
        rgb_channels = image[:, :, 0:3]
        
        # Overlay the RGB channels onto the black background using the alpha channel
        overlayed = np.zeros_like(rgb_channels)
        for c in range(0, 3):
            overlayed[:, :, c] = rgb_channels[:, :, c] * (alpha / 255.0) + \
                                 black_background[:, :, c] * (1.0 - alpha / 255.0)
        
        output_name = f"on_black_{os.path.basename(carpet_img_path)}"
        on_black_path = os.path.join(temp_path, output_name)
        cv2.imwrite(on_black_path, overlayed)
        print(f"|INFO| Image placed on black background: {on_black_path}")
        return on_black_path
    
    print("|WARNING| Input image does not have an alpha channel. Returning original path.")
    return carpet_img_path

def overlay_centered_image(background_path, foreground_path, temp_path="../Floor-Overlay/temporary"):
    """
    Overlays a foreground image onto a background image, centered on the background.
    """
    background = cv2.imread(background_path)
    h2, w2, _ = background.shape
    foreground = cv2.imread(foreground_path)
    h1, w1, _ = foreground.shape
    
    x = w2 // 2
    y = h2 // 2
    
    x1_start = x - w1 // 2
    y1_start = y - h1 // 2
    x1_end = x1_start + w1
    y1_end = y1_start + h1
    
    if x1_start < 0: 
        x1_start = 0
        x1_end = min(w1, w2)
    if y1_start < 0: 
        y1_start = 0
        y1_end = min(h1, h2)
    if x1_end > w2: 
        x1_end = w2
        x1_start = max(0, w2 - w1)
    if y1_end > h2: 
        y1_end = h2
        y1_start = max(0, h2 - h1)
    
    background[y1_start:y1_end, x1_start:x1_end] = foreground[:y1_end - y1_start, :x1_end - x1_start]
    overlayed_binary_carpet_path = os.path.join(temp_path, "overlayed_carpet.jpg")
    cv2.imwrite(overlayed_binary_carpet_path, background)
    print(f"|INFO| Image saved as {overlayed_binary_carpet_path}")
    return overlayed_binary_carpet_path