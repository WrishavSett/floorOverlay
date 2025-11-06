# utils.py

import os
import cv2
import base64
import requests
import numpy as np
from io import BytesIO

from main import *

def scale_room_image(
        room_image_path,
        temp_path="temporary",
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

def resize_for_safe_tiling(
        img,
        target_size=(800, 800)):
    """
    Resize an image to 800x800 regardless of original size.

    Args:
        img (np.ndarray): Input image.
        multiplier (int): Number of times the image will be tiled (unused now).

    Returns:
        np.ndarray: Resized image (1000x1000) safe to tile.
    """
    import cv2
    target_size = target_size
    print(f"|INFO| Resizing image to fixed size {target_size} for safe tiling.")
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def tileDesign(
        design_path,
        multiplier=3,
        temp_path="temporary"):
    print(f"|INFO| Tiling design from {design_path} with a multiplier of {multiplier}...")

    design_img = cv2.imread(design_path)
    if design_img is None:
        print(f"|ERROR| Could not read image at path: {design_path}")
        return None
    
    design_img = resize_for_safe_tiling(design_img)

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
    mask_output_dir = "../floorOverlay/mask_out"
    os.makedirs(mask_output_dir, exist_ok=True)
    mask_output_path = os.path.join(mask_output_dir, f"{room_image_name}_mask.jpg")

    # load_model()
    success = infer(room_image_path, 3, mask_output_path)
    
    if success:
        print("|INFO| Inference completed successfully. Proceeding with texture application...")
        return mask_output_path
    else:
        print("|WARNING| Feature not found in image. Exiting...")
        return None

def convert_to_binary_mask(
        room_image_path,
        temp_path="temporary"):
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
        temp_path="temporary"):
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