import os
import cv2
import base64
import uuid
import numpy as np
import requests # Import the requests library
from io import BytesIO # Import BytesIO for image data handling
from flask import Flask, request, jsonify
from flask_cors import CORS

# External imports from your modules
from overlay import overlay_carpet_trapezoid, overlay_carpet_ellipse, apply_transparency_to_black_background
from floor_mask_model import load_model, infer
from carpet_working import overlay_texture_on_floor
from mask_room_image import mask, scale_room_image, tileDesign

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for all routes
CORS(app)

# Create necessary directories if they don't already exist
for folder in ["inputRoom", "inputCarpet", "inputTile", "mask_out", "final_out", "temporary"]:
    os.makedirs(folder, exist_ok=True)

# Load the ML model once when the application starts
load_model()

# ───── Utility Functions ───────────────────────────────────── #
def decode_base64_to_image(base64_string):
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

def encode_image_to_base64(image):
    """
    Encodes an OpenCV image (numpy array) to a base64 string.
    
    Args:
        image (numpy.ndarray): The OpenCV image to encode.
        
    Returns:
        str: The base64-encoded string of the image.
    """
    _, buffer = cv2.imencode(".png", image) # Encode the image to a PNG format in memory
    return base64.b64encode(buffer).decode("utf-8") # Convert the image buffer to a base64 string

def download_image_from_url(url):
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

def get_image_from_input_data(image_input_data):
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

# ───────────────────────────────────────────────────────────── #
# ROUTES
# ───────────────────────────────────────────────────────────── #

@app.route("/ping", methods=["GET"])
def ping():
    """
    A simple health check endpoint to verify the API is running.
    
    Returns:
        tuple: A JSON response with the API status and an HTTP status code.
    """
    # Returns a simple JSON object to confirm the API is active
    return jsonify({"status": "API is live"}), 200

# ─── Carpet Overlay ─────────────────────────────────────────── #
@app.route("/overlayCarpet", methods=["POST"])
def get_transparent_carpet():
    """
    Overlays a carpet onto a room image with transparency and returns the result along with the floor mask.
    
    Returns:
        tuple: A JSON response with the processed images and a status code.
    """
    try:
        # Get data from the JSON request body
        data = request.json
        room_image_data = data.get("room_image")
        carpet_image_data = data.get("carpet_image")
        overlay_type = data.get("overlay_type", "ellipse")
        carpet_dimensions = data.get("carpet_dimensions", None)
        
        # Check for required input data
        if not room_image_data or not carpet_image_data:
            print(f"|ERROR| Missing room_image or carpet_image in request.")
            return jsonify({"error": "Both room_image and carpet_image must be provided"}), 400
        
        # Generate a unique ID for temporary file storage
        unique_id = str(uuid.uuid4())
        room_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        carpet_path = os.path.join("inputCarpet", f"carpet_{unique_id}.jpg")
        
        # Process input images, which can be base64 or URL
        room_img = get_image_from_input_data(room_image_data)
        carpet_img = get_image_from_input_data(carpet_image_data)
        
        # Save the processed images to temporary files
        cv2.imwrite(room_path, room_img)
        cv2.imwrite(carpet_path, carpet_img)
        
        print(f"|INFO| Processing carpet overlay for unique ID: {unique_id}")
        
        # Scale the room image and generate the floor mask
        scaled_room_img_path = scale_room_image(room_path)
        floor_mask_path = mask(scaled_room_img_path)
        
        # Read and encode the floor mask for the response
        floor_mask_img = cv2.imread(floor_mask_path)
        if floor_mask_img is None:
            print(f"|ERROR| Failed to read floor mask image from path: {floor_mask_path}")
            raise RuntimeError(f"Failed to read floor mask image from path: {floor_mask_path}")
        encoded_floor_mask = encode_image_to_base64(floor_mask_img)
        
        # Apply transparency to the carpet based on the chosen overlay type
        transparent_carpet_path = apply_transparency_to_black_background(
            scaled_room_img_path,
            carpet_path,
            overlay_type=overlay_type,
            carpet_dimensions=carpet_dimensions
        )
        
        if not transparent_carpet_path:
            print(f"|ERROR| Failed to generate transparent carpet for ID: {unique_id}. Check logs.")
            return jsonify({"error": "Failed to generate transparent carpet. Check logs."}), 500
        
        # Read the transparent carpet image with the alpha channel
        transparent_carpet_img = cv2.imread(transparent_carpet_path, cv2.IMREAD_UNCHANGED)
        if transparent_carpet_img is None:
            print(f"|ERROR| Failed to read transparent carpet image from path: {transparent_carpet_path}")
            raise RuntimeError(f"Failed to read transparent carpet image from path: {transparent_carpet_path}")
        
        # Encode the final transparent carpet image to base64
        encoded_transparent_carpet = encode_image_to_base64(transparent_carpet_img)
        
        print(f"|OUTPUT| Successfully generated and encoded transparent carpet for ID: {unique_id}")
        return jsonify({
            "status": "success",
            "transparent_carpet_image": encoded_transparent_carpet,
            "floor_mask_image": encoded_floor_mask
        })
    
    except Exception as e:
        print(f"|ERROR| An unexpected error occurred: {str(e)}")
        # import traceback
        # traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500

# ─── Model-Based Floor Overlay ──────────────────────────────── #
@app.route("/overlayFloor", methods=["POST"])
def overlay_floor_model():
    """
    Overlays a floor design onto a room image using a segmentation model.
    
    Returns:
        tuple: A JSON response with the final image and a status code.
    """
    try:
        # Get data from the JSON request body
        data = request.json
        room_image_data = data.get("room_image")
        design_image_data = data.get("design_image")
        
        # Check for required input data
        if not room_image_data or not design_image_data:
            print(f"|ERROR| Missing room_image or design_image in request.")
            return jsonify({"error": "Both room_image and design_image must be provided"}), 400
        
        # Generate a unique ID and file paths for temporary storage
        unique_id = str(uuid.uuid4())
        room_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        design_path = os.path.join("inputTile", f"design_{unique_id}.jpg")
        mask_path = os.path.join("mask_out", f"mask_{unique_id}.jpg")
        final_path = os.path.join("final_out", f"final_{unique_id}.jpg")
        
        # Process input images, which can be base64 or URL
        room_img = get_image_from_input_data(room_image_data)
        design_img = get_image_from_input_data(design_image_data)
        
        # Save the processed images to temporary files
        cv2.imwrite(room_path, room_img)
        cv2.imwrite(design_path, design_img)
        
        print(f"|INFO| Starting floor overlay process for ID: {unique_id}")
        
        # Scale the room image for model processing
        scaled_room_img_path = scale_room_image(room_path)
        
        # Tile the floor design image
        tiled_design_path = tileDesign(design_path)
        
        # Perform floor segmentation using the ML model
        success = infer(scaled_room_img_path, 0, mask_path)
        
        if success:
            # If segmentation is successful, overlay the tiled design onto the floor mask
            final_output = overlay_texture_on_floor(scaled_room_img_path, mask_path, tiled_design_path)
            if final_output is not None:
                # Save the final image and return the base64-encoded result
                cv2.imwrite(final_path, final_output)
                print(f"|OUTPUT| Successfully generated and encoded final floor image for ID: {unique_id}")
                return jsonify({"status": "success", "final_output": encode_image_to_base64(final_output)})
            else:
                print(f"|ERROR| Failed to generate final output for ID: {unique_id}")
                return jsonify({"error": "Failed to generate final output"}), 500
        else:
            print(f"|WARNING| Feature not found in image for ID: {unique_id}")
            return jsonify({"error": "Feature not found in image"}), 400
    
    except Exception as e:
        print(f"|ERROR| An unexpected error occurred: {str(e)}")
        # import traceback
        # traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    # Run the Flask app in debug mode on the specified host and port
    print("|INFO| Starting Flask app...")
    app.run(debug=True, host = "0.0.0.0", port = 5001)