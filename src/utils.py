import os
import cv2
import base64
import requests
import numpy as np
from io import BytesIO

from main import *

def scale_room_image(
        room_image_path,
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

def tileDesign(
        design_path,
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

def mask(
        room_image_path):
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

def convert_to_binary_mask(
        room_image_path,
        temp_path="../Floor-Overlay/temporary"):
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

def convert_to_binary_carpet(
        carpet_img_path,
        temp_path="../Floor-Overlay/temporary"):
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

def find_and_mark_floor_center(
        room_img_path,
        temp_path="../Floor-Overlay/temporary"):
    """
    Finds the centroid of the floor mask in a room image and marks it.

    Args:
        room_img_path (str): The path to the original room image.
        temp_path (str): The temporary directory to save the marked image.

    Returns:
        tuple or None: A tuple (cx, cy) of the centroid coordinates, or None if no floor mask is detected.
    """
    masked_image_path = mask(room_img_path)
    # Load the masked image
    image = cv2.imread(masked_image_path)
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for red color (adjust if needed)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Find contours of the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour assuming it's the floor
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute the centroid of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw the center point on the image
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            
            # Ensure the output folder exists
            os.makedirs(temp_path, exist_ok=True)
            
            # Save the modified image
            output_path = os.path.join(temp_path, "marked_masked_image.jpg")
            cv2.imwrite(output_path, image)

            print(f"|INFO| Marked image saved at: {output_path}")
            return (cx, cy)

    print("|WARNING| No floor mask detected.")
    return None

def scale_carpet(
        room_img_path,
        carpet_img_path,
        carpet_dimensions=None,
        temp_path="../Floor-Overlay/temporary"):
    """
    Scales a carpet image relative to a room image based on provided dimensions or a default ratio.

    Args:
        room_img_path (str): The path to the room image, used as a reference for scaling.
        carpet_img_path (str): The path to the carpet image to be scaled.
        carpet_dimensions (str, optional): The dimensions of the carpet in "width_ft/height_ft" format. Defaults to None.
        temp_path (str): The temporary directory to save the scaled image.

    Returns:
        str: The path to the saved scaled carpet image.
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
        except Exception as e:
            print(f"|WARNING| Invalid carpet_dimensions format: {carpet_dimensions}. Using default scaling.")
            max_width = max(1, ref_width // 3)
            max_height = max(1, ref_height // 3)
    else:
        max_width = max(1, ref_width // 3)
        max_height = max(1, ref_height // 3)

    img = cv2.imread(carpet_img_path)
    img_height, img_width = img.shape[:2]

    scale_factor = min(max_width / img_width, max_height / img_height)
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    output_folder = temp_path
    os.makedirs(output_folder, exist_ok=True)
    scaled_carpet_path = os.path.join(output_folder, "scaled_carpet_image.jpg")
    cv2.imwrite(scaled_carpet_path, resized_img)
    print(f"|INFO| Resized image saved as {scaled_carpet_path} with dimensions {new_width}x{new_height}")
    return scaled_carpet_path

def create_black_image(
        room_img_path,
        temp_path="../Floor-Overlay/temporary"):
    """
    Creates a black image with the same dimensions as a reference room image.

    Args:
        room_img_path (str): The path to the reference room image.
        temp_path (str): The temporary directory to save the black image.

    Returns:
        str: The path to the saved black image.
    """
    ref_img = cv2.imread(room_img_path)
    ref_height, ref_width = ref_img.shape[:2]

    black_img = np.zeros((ref_height, ref_width, 3), dtype=np.uint8)

    output_folder = temp_path
    os.makedirs(output_folder, exist_ok=True)

    black_blank_img_path = os.path.join(output_folder, "black_blank_image.jpg")

    cv2.imwrite(black_blank_img_path, black_img)
    print(f"|INFO| Black image saved as {black_blank_img_path} with dimensions {ref_width}x{ref_height}")

    return black_blank_img_path

def place_on_black(
        room_img_path,
        carpet_img_path,
        carpet_dimensions=None,
        temp_path="../Floor-Overlay/temporary"):
    """
    Places a scaled carpet image on a black background, centered on the floor mask's centroid.

    Args:
        room_img_path (str): The path to the room image.
        carpet_img_path (str): The path to the carpet image.
        carpet_dimensions (str, optional): The dimensions of the carpet for scaling. Defaults to None.
        temp_path (str): The temporary directory to save intermediate and final images.

    Returns:
        str: The path to the saved image with the carpet placed on the black background.
    """
    center_of_mask = find_and_mark_floor_center(room_img_path, temp_path)
    if not center_of_mask:
        print("|ERROR| Could not find the center of the floor mask.")
        return None
        
    x, y = center_of_mask
    background_path = create_black_image(room_img_path, temp_path)
    foreground_path = scale_carpet(room_img_path, carpet_img_path, carpet_dimensions=carpet_dimensions, temp_path=temp_path)

    background = cv2.imread(background_path)
    h2, w2, _ = background.shape
    foreground = cv2.imread(foreground_path)
    h1, w1, _ = foreground.shape

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

def adjust_carpet_perspective(
        carpet_img_path,
        temp_path="../Floor-Overlay/temporary"):
    """
    Applies a perspective transformation to a carpet image to make it look like a trapezoid.

    Args:
        carpet_img_path (str): The path to the original carpet image.
        temp_path (str): The temporary directory to save the transformed image.

    Returns:
        str: The path to the saved warped image.
    """
    image = cv2.imread(carpet_img_path)
    h, w = image.shape[:2]

    # Define the source points (corners of the original image)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Reduce the offset to achieve 110 degrees (less shrinking)
    offset = h // 3

    # Define the destination points for the new perspective
    dst_pts = np.float32([
        [offset, 0],         # Top-left (shifted inward slightly)
        [w - offset, 0],     # Top-right (shifted inward slightly)
        [w, h],              # Bottom-right (unchanged)
        [0, h]               # Bottom-left (unchanged)
    ])

    # Compute perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    warped_img_path = os.path.join(temp_path, "warped_carpet_image.jpg")
    cv2.imwrite(warped_img_path, warped)

    return warped_img_path

def apply_transparency_to_black_background(
        room_img_path,
        carpet_img_path,
        overlay_type="ellipse",
        carpet_dimensions=None,
        output_path="../Floor-Overlay/final_out",
        temp_path = "../Floor-Overlay/temporary"):
    """
    Applies transparency to a carpet image that has been placed on a black background.

    Args:
        room_img_path (str): The path to the room image, used for scaling and placement.
        carpet_img_path (str): The path to the original carpet image.
        overlay_type (str, optional): The shape of the carpet to use ('ellipse' or 'trapezoid'). Defaults to "ellipse".
        carpet_dimensions (str, optional): The dimensions of the carpet for scaling. Defaults to None.
        output_path (str): The directory to save the final transparent image.
        temp_path (str): The temporary directory for intermediate files.

    Returns:
        str or None: The path to the final transparent image, or None on failure.
    """
    carpet_on_black_path = None
    binary_carpet_mask_path = None
    room_image_name = os.path.splitext(os.path.basename(room_img_path))[0]
    
    type_abbr = ""
    if overlay_type.lower() in ["ellipse", "e"]:
        type_abbr = "e"
        print(f"|INFO| Preparing elliptical carpet for transparency...")
        elliptical_carpet_path, _ = carpet_ellipse_and_center(carpet_img_path, temp_path=temp_path)
        if not elliptical_carpet_path:
            print("|ERROR| Failed to generate elliptical carpet. Aborting transparency application.")
            return

        carpet_on_black_path = place_on_black(room_img_path, elliptical_carpet_path, carpet_dimensions=carpet_dimensions, temp_path=temp_path)
        if not carpet_on_black_path:
            print("|ERROR| Failed to place elliptical carpet on black background. Aborting transparency application.")
            return

        binary_carpet_mask_path = convert_to_binary_carpet(carpet_on_black_path, temp_path=temp_path)
        if not binary_carpet_mask_path:
            print("|ERROR| Failed to generate binary mask for elliptical carpet. Aborting transparency application.")
            return

    elif overlay_type.lower() in ["trapezoid", "t"]:
        type_abbr = "t"
        print(f"|INFO| Preparing trapezoidal carpet for transparency...")
        trapezoid_carpet_path = adjust_carpet_perspective(carpet_img_path, temp_path=temp_path)
        if not trapezoid_carpet_path:
            print("|ERROR| Failed to generate trapezoidal carpet. Aborting transparency application.")
            return

        carpet_on_black_path = place_on_black(room_img_path, trapezoid_carpet_path, carpet_dimensions=carpet_dimensions, temp_path=temp_path)
        if not carpet_on_black_path:
            print("|ERROR| Failed to place trapezoidal carpet on black background. Aborting transparency application.")
            return

        binary_carpet_mask_path = convert_to_binary_carpet(carpet_on_black_path, temp_path=temp_path)
        if not binary_carpet_mask_path:
            print("|ERROR| Failed to generate binary mask for trapezoidal carpet. Aborting transparency application.")
            return
    else:
        print(f"|ERROR| Invalid overlay_type '{overlay_type}'. Please use 'ellipse'/'e' or 'trapezoid'/'t'.")
        return

    output_filename = f"transparent_carpet_{type_abbr}_{room_image_name}.png"
    final_output_path = os.path.join(output_path, output_filename)

    print(f"|INFO| Applying transparency to the generated '{overlay_type}' carpet on black background...")
    
    carpet_on_black = cv2.imread(carpet_on_black_path)
    binary_carpet_mask = cv2.imread(binary_carpet_mask_path, cv2.IMREAD_GRAYSCALE)

    if carpet_on_black is None or binary_carpet_mask is None:
        print("|ERROR| Could not load intermediate images for transparency application.")
        return

    if len(carpet_on_black.shape) < 3 or carpet_on_black.shape[2] == 1:
        carpet_on_black = cv2.cvtColor(carpet_on_black, cv2.COLOR_GRAY2BGR)

    # Split the carpet image into BGR channels
    b, g, r = cv2.split(carpet_on_black)

    # Create a base alpha channel from the binary mask.
    alpha_base = binary_carpet_mask.copy()

    # Apply Gaussian Blur to the alpha channel to create a smooth gradient at the edges.
    blurred_alpha = cv2.GaussianBlur(alpha_base, (15, 15), 0)

    # Ensure the blurred alpha values are clamped between 0 and 255
    blurred_alpha = np.clip(blurred_alpha, 0, 255).astype(np.uint8)

    # Merge the BGR channels with the smoothed alpha channel
    transparent_image = cv2.merge([b, g, r, blurred_alpha])

    # Identify pixels that are truly black (RGB all 0) AND are part of the carpet itself
    # This prevents black patterns/details within the carpet from becoming transparent.
    carpet_pixels_mask = (binary_carpet_mask == 255)

    black_in_carpet_mask = (carpet_on_black[:, :, 0] == 0) & \
                           (carpet_on_black[:, :, 1] == 0) & \
                           (carpet_on_black[:, :, 2] == 0) & \
                           carpet_pixels_mask

    # Set the alpha channel to 255 (fully opaque) for these identified black pixels within the carpet
    transparent_image[black_in_carpet_mask, 3] = 255

    cv2.imwrite(final_output_path, transparent_image)
    print(f"|INFO| Final transparent image saved to: {final_output_path}")
    return final_output_path

def overlay_carpet_trapezoid(
        room_img_path, 
        carpet_img_path, 
        carpet_dimensions=None, 
        output_path="../Floor-Overlay/final_out"):
    """
    Overlays a trapezoidal carpet onto a room image, blending the two images based on binary masks.

    Args:
        room_img_path (str): The path to the room image.
        carpet_img_path (str): The path to the carpet image.
        carpet_dimensions (str, optional): The dimensions of the carpet for scaling. Defaults to None.
        output_path (str): The directory to save the final image.

    Returns:
        str: The path to the saved final image.
    """
    os.makedirs(output_path, exist_ok=True)
    temp_path = "../Floor-Overlay/temporary"
    os.makedirs(temp_path, exist_ok=True)

    warped_carpet_img_path = adjust_carpet_perspective(carpet_img_path, temp_path=temp_path)
    room_img = cv2.imread(room_img_path)
    if room_img is None:
        raise FileNotFoundError(f"Could not read room image at path: {room_img_path}")
    room_image_name = os.path.splitext(os.path.basename(room_img_path))[0]

    bin_mask_img_path = convert_to_binary_mask(room_img_path, temp_path=temp_path)
    bin_mask_img = cv2.imread(bin_mask_img_path, cv2.IMREAD_GRAYSCALE)
    if bin_mask_img is None:
        raise FileNotFoundError(f"Could not read binary mask image at path: {bin_mask_img_path}")

    overlayed_carpet_img_path = place_on_black(room_img_path, warped_carpet_img_path, carpet_dimensions=carpet_dimensions, temp_path=temp_path)
    overlayed_carpet_img = cv2.imread(overlayed_carpet_img_path)
    if overlayed_carpet_img is None:
        raise FileNotFoundError(f"Could not read overlayed carpet image at path: {overlayed_carpet_img_path}")

    overlayed_bin_carpet_img_path = convert_to_binary_carpet(overlayed_carpet_img_path, temp_path=temp_path)
    overlayed_bin_carpet_img = cv2.imread(overlayed_bin_carpet_img_path, cv2.IMREAD_GRAYSCALE)
    if overlayed_bin_carpet_img is None:
        raise FileNotFoundError(f"Could not read overlayed binary carpet image at path: {overlayed_bin_carpet_img_path}")

    combined_binary_mask = cv2.bitwise_and(bin_mask_img, overlayed_bin_carpet_img)
    alpha_channel = cv2.GaussianBlur(combined_binary_mask, (15, 15), 0)
    alpha_channel_normalized = alpha_channel.astype(np.float32) / 255.0
    
    alpha_channel_3d = np.stack([alpha_channel_normalized, alpha_channel_normalized, alpha_channel_normalized], axis=-1)

    room_img_float = room_img.astype(np.float32)
    overlayed_carpet_img_float = overlayed_carpet_img.astype(np.float32)

    result_float = (overlayed_carpet_img_float * alpha_channel_3d) + \
                   (room_img_float * (1 - alpha_channel_3d))

    result = np.clip(result_float, 0, 255).astype(np.uint8)

    result_img_path = os.path.join(output_path, f"overlayed_carpet_t_{room_image_name}.jpg")
    cv2.imwrite(result_img_path, result)
    return result_img_path

def overlay_carpet_ellipse(
        room_img_path,
        carpet_img_path,
        carpet_dimensions=None,
        output_path="../Floor-Overlay/final_out"):
    """
    Overlays an elliptical carpet onto a room image, blending the two images based on binary masks.

    Args:
        room_img_path (str): The path to the room image.
        carpet_img_path (str): The path to the carpet image.
        carpet_dimensions (str, optional): The dimensions of the carpet for scaling. Defaults to None.
        output_path (str): The directory to save the final image.

    Returns:
        str: The path to the saved final image.
    """
    os.makedirs(output_path, exist_ok=True)
    temp_path = "../Floor-Overlay/temporary"
    os.makedirs(temp_path, exist_ok=True)

    ellipse_carpet_path, ellipse_carpet_center = carpet_ellipse_and_center(carpet_img_path, temp_path=temp_path)
    room_img = cv2.imread(room_img_path)
    if room_img is None:
        raise FileNotFoundError(f"Could not read room image at path: {room_img_path}")
    room_image_name = os.path.splitext(os.path.basename(room_img_path))[0]

    bin_mask_img_path = convert_to_binary_mask(room_img_path, temp_path=temp_path)
    bin_mask_img = cv2.imread(bin_mask_img_path, cv2.IMREAD_GRAYSCALE)
    if bin_mask_img is None:
        raise FileNotFoundError(f"Could not read binary mask image at path: {bin_mask_img_path}")

    overlayed_carpet_img_path = place_on_black(room_img_path, ellipse_carpet_path, carpet_dimensions=carpet_dimensions, temp_path=temp_path)
    overlayed_carpet_img = cv2.imread(overlayed_carpet_img_path)
    if overlayed_carpet_img is None:
        raise FileNotFoundError(f"Could not read overlayed carpet image at path: {overlayed_carpet_img_path}")

    overlayed_bin_carpet_img_path = convert_to_binary_carpet(overlayed_carpet_img_path, temp_path=temp_path)
    overlayed_bin_carpet_img = cv2.imread(overlayed_bin_carpet_img_path, cv2.IMREAD_GRAYSCALE)
    if overlayed_bin_carpet_img is None:
        raise FileNotFoundError(f"Could not read overlayed binary carpet image at path: {overlayed_bin_carpet_img_path}")

    combined_binary_mask = cv2.bitwise_and(bin_mask_img, overlayed_bin_carpet_img)
    alpha_channel = cv2.GaussianBlur(combined_binary_mask, (15, 15), 0)
    alpha_channel_normalized = alpha_channel.astype(np.float32) / 255.0
    
    alpha_channel_3d = np.stack([alpha_channel_normalized, alpha_channel_normalized, alpha_channel_normalized], axis=-1)

    room_img_float = room_img.astype(np.float32)
    overlayed_carpet_img_float = overlayed_carpet_img.astype(np.float32)

    result_float = (overlayed_carpet_img_float * alpha_channel_3d) + \
                   (room_img_float * (1 - alpha_channel_3d))

    result = np.clip(result_float, 0, 255).astype(np.uint8)

    result_img_path = os.path.join(output_path, f"overlayed_carpet_e_{room_image_name}.jpg")
    cv2.imwrite(result_img_path, result)
    return result_img_path

def carpet_circle(
        carpet_img_path,
        temp_path="../Floor-Overlay/temporary"):
    """
    Crops a carpet image into a circle.

    Args:
        carpet_img_path (str): The path to the input carpet image.
        temp_path (str): The path to the temporary directory.

    Returns:
        str: The path to the saved circular carpet image.
    """
    # scaled_carpet_img_path = scale_carpet(room_img_path, carpet_img_path)
    # scaled_carpet_img = cv2.imread(scaled_carpet_img_path)
    
    carpet_img = cv2.imread(carpet_img_path)
    
    # Error handling for missing file
    if carpet_img is None:
        raise FileNotFoundError(f"Could not read image at path: {carpet_img_path}")
    
    height, width = carpet_img.shape[:2]

    # Add alpha channel if missing
    if carpet_img.shape[2] == 3:
        carpet_img = cv2.cvtColor(carpet_img, cv2.COLOR_BGR2BGRA)

    center = (width // 2, height // 2)
    radius = min(width, height) // 2

    # Create circular alpha mask
    circular_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circular_mask, center, radius, 255, -1)

    result = carpet_img.copy()
    result[:, :, 3] = circular_mask

    # Crop to bounding box of the circle
    x, y = center
    cropped = result[y - radius:y + radius, x - radius:x + radius]

    # Replace transparent areas with black (for JPG)
    alpha = cropped[:, :, 3]
    rgb = cropped[:, :, :3]
    mask = alpha == 0
    rgb[mask] = [0, 0, 0]  # Set transparent pixels to black

    # Final image is RGB (drop alpha)
    cropped_rgb = rgb

    # Ensure output directory exists
    os.makedirs(temp_path, exist_ok=True)

    cropped_carpet_path = os.path.join(temp_path, "carpet_circle.jpg")
    cv2.imwrite(cropped_carpet_path, cropped_rgb)
    print(f"|INFO| Circle-cropped image saved as JPG to {cropped_carpet_path}")

    return cropped_carpet_path

def carpet_ellipse_and_center(
        carpet_img_path,
        temp_path="../Floor-Overlay/temporary"):
    """
    Transforms a circular carpet image into an ellipse with a 3D perspective and finds its center.

    Args:
        carpet_img_path (str): The path to the input carpet image.
        temp_path (str): The path to the temporary directory.

    Returns:
        tuple: A tuple containing the path to the saved elliptical carpet image and a tuple representing the center coordinates (x, y).
    """
    cropped_carpet_path = carpet_circle(carpet_img_path)
    img = cv2.imread(cropped_carpet_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]

    # Parameters to control horizontal perspective distortion
    squash = height * 0.3
    shift = width * 0.2

    # Source points (original corners)
    src_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Destination points to stretch horizontally (simulate side view)
    dst_pts = np.float32([
        [shift, squash],
        [width - shift, squash],
        [width, height - squash],
        [0, height - squash]
    ])

    # Get transformation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply transformation
    warped = cv2.warpPerspective(
        img, matrix, (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Find the center of the visible ellipse (non-transparent area)
    if warped.shape[2] == 4:  # RGBA image
        alpha_channel = warped[:, :, 3]
        coords = np.column_stack(np.where(alpha_channel > 0))
        if coords.size == 0:
            center = (width // 2, height // 2)
        else:
            center_y, center_x = coords.mean(axis=0)
            center = (int(center_x), int(center_y))
    else:
        # For RGB image, use grayscale and threshold
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            center = (center_x, center_y)
        else:
            center = (width // 2, height // 2)

    print(f"|INFO| Center of the ellipse: {center}")


    # Replace transparent areas with black for JPG output
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

def decode_base64_to_image(
        base64_string):
    """
    Decodes a base64 string into an OpenCV image (numpy array).
    
    Args:
        base64_string (str): The base64-encoded string of the image.
        
    Returns:
        numpy.ndarray: The decoded image as a numpy array.
    """
    image_data = base64.b64decode(base64_string) # Decode the base64 string to raw bytes
    np_arr = np.frombuffer(image_data, np.uint8) # Convert the raw bytes to a numpy array
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Decode the numpy array into a color image using OpenCV

def encode_image_to_base64(
        image):
    """
    Encodes an OpenCV image (numpy array) to a base64 string.
    
    Args:
        image (numpy.ndarray): The OpenCV image to encode.
        
    Returns:
        str: The base64-encoded string of the image.
    """
    _, buffer = cv2.imencode(".png", image) # Encode the image to a PNG format in memory
    return base64.b64encode(buffer).decode("utf-8") # Convert the image buffer to a base64 string

def download_image_from_url(
        url):
    """
    Downloads an image from a given URL and returns it as an OpenCV image (numpy array).
    
    Args:
        url (str): The URL of the image.
        
    Returns:
        numpy.ndarray: The downloaded image as a numpy array.
    """
    try:
        response = requests.get(url, stream=True) # Send a GET request to the URL
        response.raise_for_status() # Check for bad status codes (4xx or 5xx)
        
        image_data = BytesIO(response.content) # Read the raw content of the response into an in-memory byte stream
        
        np_arr = np.frombuffer(image_data.read(), np.uint8) # Convert the byte stream to a NumPy array
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Decode the array into a color image with OpenCV
        
        if img is None:
            # Check if the image decoding was successful
            raise ValueError(f"Could not decode image from URL. It might be corrupted or not an image: {url}")
        return img
    except requests.exceptions.RequestException as e:
        # Handle specific request-related exceptions like network issues or bad URLs
        raise ConnectionError(f"Failed to download image from URL {url} due to a request error: {e}")
    except Exception as e:
        # Handle any other unexpected exceptions during the process
        raise RuntimeError(f"An unexpected error occurred while processing image from URL {url}: {e}")

def get_image_from_input_data(
        image_input_data):
    """
    Determines if input data is a URL or base64 string and processes it accordingly.
    
    Args:
        image_input_data (str): The image data, either a URL or a base64 string.
        
    Returns:
        numpy.ndarray: The processed image as a numpy array.
    """
    # Check if the input string is a URL (starts with "http" or "https")
    if image_input_data.startswith("http://") or image_input_data.startswith("https://"):
        return download_image_from_url(image_input_data)
    else:
        # Otherwise, assume it's a base64 string
        return decode_base64_to_image(image_input_data)
    
def image_to_base64(
        image_path):
    """
    Encodes an image file to a base64 string.
    
    Args:
        image_path (str): The path to the input image file.
        
    Returns:
        str: The base64-encoded string of the image.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def save_base64_image(
        base64_string,
        filename):
    """
    Saves a base64-encoded string as an image file.
    
    Args:
        base64_string (str): The base64 string of the image.
        filename (str): The path to save the output image.
        
    Returns:
        None
    """
    image_data = base64.b64decode(base64_string)
    with open(filename, "wb") as file:
        file.write(image_data)