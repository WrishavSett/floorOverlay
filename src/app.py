import os
import cv2
import base64
import uuid
import numpy as np
import requests # Import the requests library
from io import BytesIO # Import BytesIO for image data handling
from flask import Flask, request, jsonify
from flask_cors import CORS

# External imports from your new modules
from main import overlay_carpet_trapezoid, overlay_carpet_ellipse, apply_transparency_to_black_background
from main import load_model, infer
from main import overlay_texture_on_floor
from main import mask
from utils import decode_base64_to_image, encode_image_to_base64
from utils import scale_room_image, tileDesign

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for all routes
CORS(app)

# Create necessary directories if they don't already exist
for folder in ["inputRoom", "inputCarpet", "inputTile", "mask_out", "final_out", "temporary"]:
    os.makedirs(folder, exist_ok=True)

# Load the ML model once when the application starts
load_model()

# ───── API Endpoints ───────────────────────────────────── #

@app.route('/overlay_carpet_trapezoid', methods=['POST'])
def process_carpet_overlay():
    try:
        data = request.get_json()
        room_b64 = data['room_image']
        carpet_b64 = data['carpet_image']
        overlay_type = data.get('overlay_type', 'trapezoid')
        
        # Decode base64 images
        room_image = decode_base64_to_image(room_b64)
        carpet_image = decode_base64_to_image(carpet_b64)
        
        # Generate a unique ID for temporary file handling
        unique_id = str(uuid.uuid4())
        room_path = os.path.join('temporary', f'room_{unique_id}.jpg')
        carpet_path = os.path.join('temporary', f'carpet_{unique_id}.jpg')
        
        # Save temporary files
        cv2.imwrite(room_path, room_image)
        cv2.imwrite(carpet_path, carpet_image)
        
        # Perform overlay based on type
        if overlay_type == 'trapezoid':
            result_path = overlay_carpet_trapezoid(room_path, carpet_path)
        elif overlay_type == 'ellipse':
            result_path = overlay_carpet_ellipse(room_path, carpet_path)
        elif overlay_type == 'transparent':
            result_path = apply_transparency_to_black_background(room_path, carpet_path)
        else:
            return jsonify({"error": "Invalid overlay_type"}), 400
        
        # Read the result image and encode to base64
        result_img = cv2.imread(result_path)
        if result_img is None:
            return jsonify({"error": "Failed to generate result image"}), 500
            
        base64_output = encode_image_to_base64(result_img)
        
        # Cleanup temporary files
        os.remove(room_path)
        os.remove(carpet_path)
        os.remove(result_path)
        
        return jsonify({"status": "success", "transparent_carpet_image": base64_output})
    
    except Exception as e:
        print(f"|ERROR| An unexpected error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/transform_carpet', methods=['POST'])
def process_transform_carpet():
    try:
        data = request.get_json()
        carpet_b64 = data['carpet_image']

        unique_id = str(uuid.uuid4())
        carpet_path = os.path.join('temporary', f'carpet_transform_{unique_id}.jpg')

        carpet_image = decode_base64_to_image(carpet_b64)
        cv2.imwrite(carpet_path, carpet_image)

        warped_path = adjust_carpet_perspective(carpet_path)

        warped_image = cv2.imread(warped_path)
        base64_output = encode_image_to_base64(warped_image)
        
        os.remove(carpet_path)
        os.remove(warped_path)

        return jsonify({"status": "success", "transformed_carpet": base64_output})
        
    except Exception as e:
        print(f"|ERROR| An unexpected error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/overlay_floor_model', methods=['POST'])
def process_floor_overlay_model():
    unique_id = str(uuid.uuid4())
    
    try:
        data = request.get_json()
        room_b64 = data['room_image']
        design_b64 = data['design_image']
        
        # Paths for temporary files
        room_path = os.path.join('inputRoom', f'room_{unique_id}.jpg')
        mask_path = os.path.join('mask_out', f'mask_{unique_id}.jpg')
        final_path = os.path.join('final_out', f'final_{unique_id}.jpg')
        design_path = os.path.join('inputTile', f'design_{unique_id}.jpg')
        
        # Decode and save the room image
        room_image = decode_base64_to_image(room_b64)
        if room_image is None:
            return jsonify({"error": "Invalid room image"}), 400
        cv2.imwrite(room_path, room_image)
        print(f"|INFO| Saved room image to {room_path}")
        
        # Decode and save the design image
        design_image = decode_base64_to_image(design_b64)
        if design_image is None:
            return jsonify({"error": "Invalid design image"}), 400
        cv2.imwrite(design_path, design_image)
        print(f"|INFO| Saved design image to {design_path}")
        
        # Scale the room image and tile the design
        scaled_room_img_path = scale_room_image(room_path)
        tiled_design_path = tileDesign(design_path)
        
        # Perform inference using the ML model
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
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    # Run the Flask app in debug mode on the specified host and port
    app.run(debug=True, host='0.0.0.0', port=5001)