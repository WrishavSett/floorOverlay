# 016

import os
import cv2
import numpy as np
import pandas as pd
# from scale_and_overlay import scale_carpet

def carpet_circle(carpet_img_path, temp_path="../floorOverlay/temporary"):
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
    print(f"016 Circle-cropped image saved as JPG to {cropped_carpet_path}")

    return cropped_carpet_path

def carpet_ellipse_and_center(carpet_img_path, temp_path="../floorOverlay/temporary"):
    cropped_carpet_path = carpet_circle(carpet_img_path)
    img = cv2.imread(cropped_carpet_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]

    # Parameters to control horizontal perspective distortion
    squash = height * 0.3  # How much to push top and bottom inward
    shift = width * 0.2    # Optional: adds a slight lean for realism

    # Source points (original corners)
    src_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Destination points to stretch horizontally (simulate side view)
    dst_pts = np.float32([
        [shift, squash],                          # top-left
        [width - shift, squash],                 # top-right
        [width, height - squash],                # bottom-right
        [0, height - squash]                     # bottom-left
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
            center = (width // 2, height // 2)  # fallback
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
            center = (width // 2, height // 2)  # fallback

    print(f"016 Center of the ellipse: {center}")


    # output_name = "carpet_ellipse.png"
    # carpet_ellipse_path = os.path.join(temp_path, output_name)
    # cv2.imwrite(carpet_ellipse_path, warped)
    # print(f"016 Horizontally-stretched 3D perspective carpet saved to {carpet_ellipse_path}")

    # Replace transparent areas with black for JPG output
    if warped.shape[2] == 4:
        alpha_channel = warped[:, :, 3]
        rgb_channels = warped[:, :, :3]
        mask = alpha_channel == 0
        rgb_channels[mask] = [0, 0, 0]  # Set transparent pixels to black
        warped = rgb_channels  # Drop alpha

    output_name = "carpet_ellipse.jpg"
    carpet_ellipse_path = os.path.join(temp_path, output_name)
    cv2.imwrite(carpet_ellipse_path, warped)
    print(f"016 Horizontally-stretched 3D perspective carpet saved to {carpet_ellipse_path}")

    return carpet_ellipse_path, center

def main():
    carpet_ellipse_and_center("../floorOverlay/inputCarpet/carpet2.jpg")

if __name__ == "__main__":
    main()