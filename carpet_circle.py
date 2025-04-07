# 016

import os
import cv2
import numpy as np
import pandas as pd

def carpet_circle(carpet_img_path, temp="../floorOverlay/temporary"):
    carpet_img = cv2.imread(carpet_img_path, cv2.IMREAD_UNCHANGED)
    height, width = carpet_img.shape[:2]

    # Add alpha channel if it doesn't exist
    if carpet_img.shape[2] == 3:
        carpet_img = cv2.cvtColor(carpet_img, cv2.COLOR_BGR2BGRA)

    # Default center is image center
    center = (width // 2, height // 2)

    # Default radius is smallest distance to image edge
    radius = min(width, height) // 2

    # Create circular mask
    circular_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circular_mask, center, radius, 255, -1)

    # Apply circular circular_mask to alpha channel
    result = carpet_img.copy()
    result[:, :, 3] = circular_mask

    # Crop to bounding box of the circle
    x, y = center
    cropped = result[y - radius:y + radius, x - radius:x + radius]

    # Make sure output directory exists
    cropped_carpet_path = os.path.join(temp, "carpet_circle.png")

    # Save image as PNG with transparency
    cv2.imwrite(cropped_carpet_path, cropped)
    print(f"Circle cropped image saved to {temp}")

    return cropped_carpet_path

# # Based off of scale in 2D
# def carpet_ellipse(carpet_img_path, temp="../floorOverlay/temporary"):
#     cropped_carpet_path = carpet_circle(carpet_img_path)
#     img = cv2.imread(cropped_carpet_path, cv2.IMREAD_UNCHANGED)
#     height, width = img.shape[:2]
#     scale_y=0.5

#     # Define source points (corners of original square image)
#     src_pts = np.float32([
#         [0, 0],
#         [width, 0],
#         [width, height],
#         [0, height]
#     ])

#     # Define destination points to squash vertically (simulate perspective)
#     dst_pts = np.float32([
#         [0, height * (1 - scale_y) / 2],
#         [width, height * (1 - scale_y) / 2],
#         [width, height * (1 + scale_y) / 2],
#         [0, height * (1 + scale_y) / 2]
#     ])

#     # Get perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     # Apply the warp
#     warped = cv2.warpPerspective(img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

#     output_name="carpet_ellipse.png"
#     output_path = os.path.join(temp, output_name)
#     cv2.imwrite(output_path, warped)
#     print(f"Elliptical perspective image saved to {output_path}")
#     return output_path

# # Based off of 3D Horizontal
# def carpet_ellipse(carpet_img_path, temp="../floorOverlay/temporary"):
#     cropped_carpet_path = carpet_circle(carpet_img_path)
#     img = cv2.imread(cropped_carpet_path, cv2.IMREAD_UNCHANGED)
#     height, width = img.shape[:2]

#     # Parameters for 3D perspective feel
#     shrink = width * 0.3  # Controls how narrow the top appears
#     lift = height * 0.2   # Simulates camera tilt by lifting the top edge

#     # Source points (corners of the original image)
#     src_pts = np.float32([
#         [0, 0],
#         [width, 0],
#         [width, height],
#         [0, height]
#     ])

#     # Destination points for 3D perspective warping
#     dst_pts = np.float32([
#         [shrink, lift],                     # top-left
#         [width - shrink, lift],            # top-right
#         [width, height],                   # bottom-right
#         [0, height]                        # bottom-left
#     ])

#     # Compute the perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     # Apply the warp with transparency preserved
#     warped = cv2.warpPerspective(
#         img, matrix, (width, height),
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(0, 0, 0, 0)
#     )

#     output_name = "carpet_ellipse.png"
#     output_path = os.path.join(temp, output_name)
#     cv2.imwrite(output_path, warped)
#     print(f"3D perspective elliptical carpet saved to {output_path}")
#     return output_path

def carpet_ellipse(carpet_img_path, temp="../floorOverlay/temporary"):
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

    output_name = "carpet_ellipse.png"
    output_path = os.path.join(temp, output_name)
    cv2.imwrite(output_path, warped)
    print(f"Horizontally-stretched 3D perspective carpet saved to {output_path}")
    return output_path

def main():
    carpet_ellipse("../floorOverlay/carpet/carpet2.jpg")

if __name__ == "__main__":
    main()