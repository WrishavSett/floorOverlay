# app.py

import os
import cv2
import uuid
import base64
import requests
import numpy as np
from io import BytesIO
from flask_cors import CORS
from flask import Flask, request, jsonify

# External imports from your modules
from utils import *
from main import *

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for all routes
CORS(app)

# Create necessary directories if they don't already exist
for folder in ["inputRoom",
               "inputCarpet",
               "inputTile",
               "mask_out",
               "final_out",
               "temporary"]:
    os.makedirs(folder, exist_ok=True)

# Load the ML model once when the application starts
load_model()

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
def overlay_carpet():
    """
    Overlays a carpet onto a room image based on user-provided dimensions.
    
    Returns:
        tuple: A JSON response with the final processed image and a status code.
    """
    try:
        # Get data from the JSON request body
        data = request.json
        room_image_data = data.get("room_image")
        carpet_image_data = data.get("carpet_image")
        room_dimensions_str = data.get("room_dimensions")
        carpet_dimensions_str = data.get("carpet_dimensions")
        room_dim_unit = data.get("room_dim_unit", "ft")
        carpet_dim_unit = data.get("carpet_dim_unit", "ft")
        
        # Check for required input data
        if not all([room_image_data, carpet_image_data, room_dimensions_str, carpet_dimensions_str]):
            print(f"|ERROR| Missing required data in request.")
            return jsonify({"error": "room_image, carpet_image, room_dimensions, and carpet_dimensions must be provided"}), 400

        # Convert dimensions to a common unit (feet)
        def convert_to_feet(value_str, unit):
            width, height = map(float, value_str.split('/'))
            if unit.lower() == 'm':
                width *= 3.28084
                height *= 3.28084
            elif unit.lower() == 'in':
                width /= 12.0
                height /= 12.0
            return f"{width}/{height}"

        converted_room_dimensions = convert_to_feet(room_dimensions_str, room_dim_unit)
        converted_carpet_dimensions = convert_to_feet(carpet_dimensions_str, carpet_dim_unit)
        
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
        mask_path = os.path.join("mask_out", f"mask_{unique_id}.jpg")
        success = infer(scaled_room_img_path, 3, mask_path) # Use mode 3 for floor

        if not success:
            print(f"|WARNING| Floor not found in image for ID: {unique_id}")
            return jsonify({"error": "Floor not found in image"}), 400

        # Read and encode the floor mask for the response
        floor_mask_img = cv2.imread(mask_path)
        if floor_mask_img is None:
            print(f"|ERROR| Failed to read floor mask image from path: {mask_path}")
            raise RuntimeError(f"Failed to read floor mask image from path: {mask_path}")
        encoded_floor_mask = encode_image_to_base64(floor_mask_img)

        # Call the new overlay function
        warped_carpet_output = overlay_carpet_on_room(
            scaled_room_img_path,
            carpet_path,
            converted_room_dimensions,
            converted_carpet_dimensions,
            mask_path
        )

        if warped_carpet_output is None:
            print(f"|ERROR| Failed to generate warped carpet output for ID: {unique_id}")
            return jsonify({"error": "Failed to generate warped carpet output"}), 500

        print(f"|OUTPUT| Successfully generated and encoded warped carpet image for ID: {unique_id}")
        return jsonify({"status": "success", "carpet_image": encode_image_to_base64(warped_carpet_output), "floor_mask_image": encoded_floor_mask})
    
    except Exception as e:
        print(f"|ERROR| An unexpected error occurred: {str(e)}")
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
        success = infer(scaled_room_img_path, 3, mask_path)
        
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