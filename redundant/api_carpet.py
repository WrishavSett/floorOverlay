#017

import os
import cv2
import base64
import uuid
import numpy as np
from flask import Flask, request, jsonify
from overlay import overlay_carpet_trapezoid, overlay_carpet_ellipse

app = Flask(__name__)

# Create necessary folders
os.makedirs("inputRoom", exist_ok=True)
os.makedirs("inputCarpet", exist_ok=True)
os.makedirs("temporary", exist_ok=True)
os.makedirs("final_out", exist_ok=True)

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
    return jsonify({"status": "Carpet Overlay API is live"}), 200

@app.route("/overlayCarpet", methods=["POST"])
def overlay_carpet():
    try:
        data = request.json
        room_image_b64 = data.get("room_image")
        carpet_image_b64 = data.get("carpet_image")
        overlay_type = data.get("overlay_type", "trapezoid")  # default to trapezoid

        if not room_image_b64 or not carpet_image_b64:
            return jsonify({"error": "Both room_image and carpet_image must be provided"}), 400

        # Generate a unique identifier
        unique_id = str(uuid.uuid4())

        # Define file paths
        room_image_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        carpet_image_path = os.path.join("inputCarpet", f"carpet_{unique_id}.jpg")
        final_output_path = os.path.join("final_out", f"final_{unique_id}.jpg")

        # Decode and save input images
        room_image = decode_base64_to_image(room_image_b64)
        carpet_image = decode_base64_to_image(carpet_image_b64)
        cv2.imwrite(room_image_path, room_image)
        cv2.imwrite(carpet_image_path, carpet_image)

        # Apply chosen overlay
        if overlay_type == "ellipse":
            result_path = overlay_carpet_ellipse(room_image_path, carpet_image_path, output_path="final_out")
        else:
            result_path = overlay_carpet_trapezoid(room_image_path, carpet_image_path, output_path="final_out")

        # Read and encode the result
        result_image = cv2.imread(result_path)
        result_b64 = encode_image_to_base64(result_image)

        return jsonify({"status": "success", "final_output": result_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


# POST /overlay
# {
#   "room_image": "<base64_encoded_image>",
#   "carpet_image": "<base64_encoded_image>",
#   "overlay_type": "ellipse"  // or "trapezoid"
# }