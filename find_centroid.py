# 013

import cv2
import numpy as np
import os
from mask_room_image import mask

def find_and_mark_floor_center(room_img_path, temp_path="../floorOverlay/temporary"):
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
    red_mask = mask1 + mask2  # Combine both masks

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
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Green circle
            
            # Ensure the output folder exists
            os.makedirs(temp_path, exist_ok=True)
            
            # Save the modified image
            output_path = os.path.join(temp_path, "marked_masked_image.jpg")
            cv2.imwrite(output_path, image)

            print(f"013 Marked image saved at: {output_path}")
            return (cx, cy)

    print("013 No floor mask detected.")
    return None

def main():
    # Example usage
    room_image_path = "../floorOverlay/inputRoom/room4.jpg"
    center_point = find_and_mark_floor_center(room_image_path)

    if center_point:
        print(f"013 Center of the floor mask: {center_point}")
    else:
        print("013 Could not determine the center.")

if __name__ == "__main__":
    main()