# 004

import os
import cv2
import base64
import uuid
import numpy as np
from flask import Flask, request, jsonify
from floor_mask_model import load_model, infer
from carpet_working import overlay_texture_on_floor

app = Flask(__name__)

# Ensure directories exist
os.makedirs("inputRoom", exist_ok=True)
os.makedirs("inputTile", exist_ok=True)
os.makedirs("mask_out", exist_ok=True)
os.makedirs("final_out", exist_ok=True)

# Load the model once on startup
load_model()

def decode_base64_to_image(base64_string):
    """Convert a base64 string to an OpenCV image."""
    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image_to_base64(image):
    """Convert an OpenCV image to a base64 string."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "Floor Overlay API is live"}), 200

@app.route("/overlayFloor", methods=["POST"])
def process_images():
    try:
        data = request.json
        room_image_b64 = data.get("room_image")
        design_image_b64 = data.get("design_image")

        if not room_image_b64 or not design_image_b64:
            return jsonify({"error": "Both room_image and design_image must be provided"}), 400

        # Generate a unique identifier for this request
        unique_id = str(uuid.uuid4())

        # Define file paths with unique names
        room_image_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        design_image_path = os.path.join("inputTile", f"design_{unique_id}.jpg")
        mask_output_path = os.path.join("mask_out", f"mask_{unique_id}.jpg")
        final_output_path = os.path.join("final_out", f"final_{unique_id}.jpg")

        # Convert base64 to images and save them
        room_image = decode_base64_to_image(room_image_b64)
        design_image = decode_base64_to_image(design_image_b64)

        cv2.imwrite(room_image_path, room_image)
        cv2.imwrite(design_image_path, design_image)

        print(f"Room image saved at: {room_image_path}")
        print(f"Design image saved at: {design_image_path}")

        # Perform inference
        success = infer(room_image_path, 0, mask_output_path)

        if success:
            final_output = overlay_texture_on_floor(room_image_path, mask_output_path, design_image_path)

            if final_output is not None:
                # Save final output image
                cv2.imwrite(final_output_path, final_output)
                print(f"Final output saved at: {final_output_path}")

                # Convert final output to base64
                final_output_b64 = encode_image_to_base64(final_output)

                return jsonify({"status": "success", "final_output": final_output_b64})
            else:
                return jsonify({"error": "Failed to generate final output"}), 500
        else:
            return jsonify({"error": "Feature not found in image"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
