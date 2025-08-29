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

app = Flask(__name__)
CORS(app)
# Create necessary directories
for folder in ["inputRoom", "inputCarpet", "inputTile", "mask_out", "final_out", "temporary"]:
    os.makedirs(folder, exist_ok=True)

# Load ML model once at startup
load_model()

# Utils
def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")

# NEW UTILITY: Function to download image from a URL
def download_image_from_url(url):
    """
    Downloads an image from a given URL and returns it as an OpenCV image (numpy array).
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Read the image data from the response content
        image_data = BytesIO(response.content)
        
        # Convert image data to a numpy array and then decode with OpenCV
        np_arr = np.frombuffer(image_data.read(), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Could not decode image from URL. It might be corrupted or not an image: {url}")
        return img
    except requests.exceptions.RequestException as e:
        # Catch specific requests errors (e.g., network issues, invalid URL, timeouts)
        raise ConnectionError(f"Failed to download image from URL {url} due to a request error: {e}")
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise RuntimeError(f"An unexpected error occurred while processing image from URL {url}: {e}")

# Helper to process image data (either base64 or URL)
def get_image_from_input_data(image_input_data):
    if image_input_data.startswith("http://") or image_input_data.startswith("https://"):
        return download_image_from_url(image_input_data)
    else:
        return decode_base64_to_image(image_input_data)

# ───────────────────────────────────────────────────────────── #
# ROUTES
# ───────────────────────────────────────────────────────────── #

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "API is live"}), 200

# ─── Carpet Overlay ─────────────────────────────────────────── #
@app.route("/overlayCarpet", methods=["POST"])
def get_transparent_carpet():
    try:
        data = request.json
        room_image_data = data.get("room_image")    # Can be base64 or URL
        carpet_image_data = data.get("carpet_image") # Can be base64 or URL
        overlay_type = data.get("overlay_type", "ellipse")
        carpet_dimensions = data.get("carpet_dimensions", None)

        if not room_image_data or not carpet_image_data:
            return jsonify({"error": "Both room_image and carpet_image must be provided"}), 400

        unique_id = str(uuid.uuid4())
        room_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        carpet_path = os.path.join("inputCarpet", f"carpet_{unique_id}.jpg")

        # Process input images
        room_img = get_image_from_input_data(room_image_data)
        carpet_img = get_image_from_input_data(carpet_image_data)

        cv2.imwrite(room_path, room_img)
        cv2.imwrite(carpet_path, carpet_img)

        # ----------------- ADDITION FOR FLOOR MASK -----------------
        # Step 1: Generate the floor mask using the room image path
        # Applying scaling up/down right after user input to avoid multiple changes/repetetive function calls
        scaled_room_img_path = scale_room_image(room_path)
        room_path = scaled_room_img_path
        floor_mask_path = mask(room_path)
        
        # Step 2: Read the floor mask image
        floor_mask_img = cv2.imread(floor_mask_path)
        if floor_mask_img is None:
            raise RuntimeError(f"Failed to read floor mask image from path: {floor_mask_path}")
        
        # Step 3: Encode the floor mask image to base64
        encoded_floor_mask = encode_image_to_base64(floor_mask_img)
        # -------------------------------------------------------------

        transparent_carpet_path = apply_transparency_to_black_background(
            room_path,
            carpet_path,
            overlay_type=overlay_type,
            carpet_dimensions=carpet_dimensions
        )

        if not transparent_carpet_path:
            return jsonify({"error": "Failed to generate transparent carpet. Check logs."}), 500

        # IMREAD_UNCHANGED is important for transparent images (alpha channel)
        transparent_carpet_img = cv2.imread(transparent_carpet_path, cv2.IMREAD_UNCHANGED)
        if transparent_carpet_img is None: # Added check for successful image read
            raise RuntimeError(f"Failed to read transparent carpet image from path: {transparent_carpet_path}")
        
        # encoded_room_img = encode_image_to_base64(room_img) # CHANGED: Removed original room image
        encoded_transparent_carpet = encode_image_to_base64(transparent_carpet_img)

        return jsonify({
            "status": "success",
            # "original_room_image": encoded_room_img, # CHANGED: Removed original room image
            "transparent_carpet_image": encoded_transparent_carpet,
            "floor_mask_image": encoded_floor_mask # CHANGED: Added floor mask image
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─── Model-Based Floor Overlay ──────────────────────────────── #
@app.route("/overlayFloor", methods=["POST"])
def overlay_floor_model():
    try:
        data = request.json
        room_image_data = data.get("room_image")     # Can be base64 or URL
        design_image_data = data.get("design_image") # Can be base64 or URL

        if not room_image_data or not design_image_data:
            return jsonify({"error": "Both room_image and design_image must be provided"}), 400

        unique_id = str(uuid.uuid4())
        room_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        design_path = os.path.join("inputTile", f"design_{unique_id}.jpg")
        mask_path = os.path.join("mask_out", f"mask_{unique_id}.jpg")
        final_path = os.path.join("final_out", f"final_{unique_id}.jpg")

        # Process input images
        room_img = get_image_from_input_data(room_image_data)
        design_img = get_image_from_input_data(design_image_data)

        cv2.imwrite(room_path, room_img)
        cv2.imwrite(design_path, design_img)

        # Applying scaling up/down right after user input to avoid multiple changes/repetetive function calls
        scaled_room_img_path = scale_room_image(room_path)
        room_path = scaled_room_img_path

        # Applying tiling to the floor design image
        tiled_design_path = tileDesign(design_path)
        design_path = tiled_design_path

        success = infer(room_path, 0, mask_path)

        if success:
            final_output = overlay_texture_on_floor(room_path, mask_path, design_path)
            if final_output is not None:
                cv2.imwrite(final_path, final_output)
                return jsonify({"status": "success", "final_output": encode_image_to_base64(final_output)})
            else:
                return jsonify({"error": "Failed to generate final output"}), 500
        else:
            return jsonify({"error": "Feature not found in image"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    #flask app

if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0", port = 5001)
