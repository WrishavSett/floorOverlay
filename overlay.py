# 015

import os
import cv2
import numpy as np
from scale_and_overlay import place_on_black
from convert_binary import convert_to_binary_mask, convert_to_binary_carpet
from carpet_circle import carpet_ellipse_and_center

def adjust_carpet_perspective(carpet_img_path, temp_path="../Floor-Overlay/temporary"):
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

def overlay_carpet_trapezoid(room_img_path, carpet_img_path, carpet_dimensions=None, output_path="../Floor-Overlay/final_out"):
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

def overlay_carpet_ellipse(room_img_path, carpet_img_path, carpet_dimensions=None, output_path="../Floor-Overlay/final_out"):
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

def main():
    """
    Main function to demonstrate the transparency application for both elliptical and trapezoidal carpets.
    """
    room_img_path = "../Floor-Overlay/sample_images2/rooms/room1.jpg"
    carpet_img_path = "../Floor-Overlay/sample_images2/carpets/carpet2.jpg"
    
    # Example usage for elliptical carpet
    result_path_e = apply_transparency_to_black_background(
        room_img_path,
        carpet_img_path,
        overlay_type="ellipse",
        carpet_dimensions="13/9"
    )
    if result_path_e:
        print(f"|OUTPUT| Elliptical carpet result saved to: {result_path_e}")
    
    # Example usage for trapezoidal carpet
    result_path_t = apply_transparency_to_black_background(
        room_img_path,
        carpet_img_path,
        overlay_type="trapezoid",
        carpet_dimensions="13/9"
    )
    if result_path_t:
        print(f"|OUTPUT| Trapezoidal carpet result saved to: {result_path_t}")

if __name__ == "__main__":
    from floor_mask_model import load_model
    load_model()
    main()