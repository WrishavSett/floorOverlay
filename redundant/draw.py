# Redundant as of now
# Refer to find_centroid.py instead

import cv2
import numpy as np
from mask_room_image import mask

def extract_red_mask(image_path):
    """Extracts the red mask from the input image."""
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])

    # Create binary mask for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    return mask1 + mask2  # Combined red mask

def find_largest_contour(mask):
    """Finds the largest contour in the mask and returns a binary mask of it."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_mask = np.zeros_like(mask)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return largest_mask

def compute_y_coordinates(mask):
    """Finds key y-coordinates in the largest contour mask."""
    ys, xs = np.where(mask == 255)  # Get white pixel coordinates

    if len(ys) == 0:  # If no white pixels found
        return None

    y_min, y_max = np.min(ys), np.max(ys)
    segment_height = (y_max - y_min) // 3

    y1_end = y_min + segment_height  # End of Part 1
    y3_start = y_max - segment_height  # Start of Part 3

    # Find max y in Part 1
    part1_mask = ys[ys <= y1_end]
    max_y_part1 = np.max(part1_mask) if len(part1_mask) > 0 else y_min

    # Find min y in Part 3
    part3_mask = ys[ys >= y3_start]
    min_y_part3 = np.min(part3_mask) if len(part3_mask) > 0 else y_max

    # Compute center y-coordinate
    center_y = (max_y_part1 + min_y_part3) // 2
    center_x = mask.shape[1] // 2  # Image center horizontally

    return y_min, y_max, max_y_part1, min_y_part3, center_x, center_y

def annotate_and_save(mask, y_min, y_max, max_y_part1, min_y_part3, center_x, center_y, output_path):
    """Draws markings and saves the final annotated image."""
    output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw horizontal lines for topmost & bottommost points
    cv2.line(output_image, (0, y_min), (output_image.shape[1], y_min), (255, 0, 0), 2)  # Topmost
    cv2.line(output_image, (0, y_max), (output_image.shape[1], y_max), (255, 0, 0), 2)  # Bottommost

    # Draw horizontal lines for segment separators
    segment_height = (y_max - y_min) // 3
    y1_end = y_min + segment_height
    y3_start = y_max - segment_height
    cv2.line(output_image, (0, y1_end), (output_image.shape[1], y1_end), (0, 255, 255), 2)  # End of Part 1
    cv2.line(output_image, (0, y3_start), (output_image.shape[1], y3_start), (0, 255, 255), 2)  # Start of Part 3

    # Mark key points
    cv2.circle(output_image, (center_x, max_y_part1), 5, (0, 255, 0), -1)  # Max Y in Part 1 (Green)
    cv2.circle(output_image, (center_x, min_y_part3), 5, (0, 0, 255), -1)  # Min Y in Part 3 (Red)
    cv2.circle(output_image, (center_x, center_y), 5, (255, 0, 255), -1)  # Center Point (Purple)

    # Save the annotated image
    cv2.imwrite(output_path, output_image)
    print(f"Annotated image saved as '{output_path}'")

def process_image(image_path, largest_contour_mask, marked_contour_mask):
    """Main function to process the image and generate the required outputs."""
    # Step 1: Extract red mask
    red_mask = extract_red_mask(image_path)

    # Step 2: Find the largest contour and create a new mask
    largest_mask = find_largest_contour(red_mask)
    cv2.imwrite(largest_contour_mask, largest_mask)  # Save mask
    print(f"Largest contour mask saved as '{largest_contour_mask}'")

    # Step 3: Compute key y-coordinates
    y_coords = compute_y_coordinates(largest_mask)
    if y_coords is None:
        print("No valid contours found.")
        return

    y_min, y_max, max_y_part1, min_y_part3, center_x, center_y = y_coords
    # print(f"Topmost y: {y_min}, Bottommost y: {y_max}")
    # print(f"Max y in Part 1: {max_y_part1}, Min y in Part 3: {min_y_part3}")
    # print(f"Center Point: ({center_x}, {center_y})")

    topmost_y, bottommost_y = y_min, y_max
    p1_max_y, p3_min_y = max_y_part1, min_y_part3
    center_p2 = (center_x, center_y)

    # Step 4: Annotate the mask and save
    annotate_and_save(largest_mask, y_min, y_max, max_y_part1, min_y_part3, center_x, center_y, marked_contour_mask)
    return topmost_y, bottommost_y, p1_max_y, p3_min_y, center_p2

def main():
    mask_path = mask(room_image_path="../floorOverlay/inputRoom/room4.jpg")
    # Run the function with file paths
    topmost_y, bottommost_y, p1_max_y, p3_min_y, center_p2 = process_image(
        image_path=mask_path,
        largest_contour_mask='../floorOverlay/temporary/largest_contour_mask.jpg',
        marked_contour_mask='../floorOverlay/temporary/marked_contour_mask.jpg'
    )
    print(f"Topmost y: {topmost_y}, Bottommost y: {bottommost_y}")
    print(f"Max y in Part 1: {p1_max_y}, Min y in Part 3: {p3_min_y}")
    print(f"Center Point: {center_p2}")

if __name__ == "__main__":
    main()