import os
import cv2
import base64
import uuid
import numpy as np
from flask import Flask, request, jsonify

# External imports from your modules
from overlay import overlay_carpet_trapezoid, overlay_carpet_ellipse
from floor_mask_model import load_model, infer
from carpet_working import overlay_texture_on_floor
# from floor_overlay import overlay

app = Flask(__name__)

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
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# ───────────────────────────────────────────────────────────── #
# ROUTES
# ───────────────────────────────────────────────────────────── #

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "API is live"}), 200

# ─── Carpet Overlay ─────────────────────────────────────────── #
@app.route("/overlayCarpet", methods=["POST"])
def overlay_carpet():
    try:
        data = request.json
        room_image_b64 = data.get("room_image")
        carpet_image_b64 = data.get("carpet_image")
        overlay_type = data.get("overlay_type", "ellipse")

        if not room_image_b64 or not carpet_image_b64:
            return jsonify({"error": "Both room_image and carpet_image must be provided"}), 400

        unique_id = str(uuid.uuid4())
        room_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        carpet_path = os.path.join("inputCarpet", f"carpet_{unique_id}.jpg")

        room_img = decode_base64_to_image(room_image_b64)
        carpet_img = decode_base64_to_image(carpet_image_b64)

        cv2.imwrite(room_path, room_img)
        cv2.imwrite(carpet_path, carpet_img)

        if overlay_type == "ellipse":
            result_path = overlay_carpet_ellipse(room_path, carpet_path, output_path="final_out")
        else:
            result_path = overlay_carpet_trapezoid(room_path, carpet_path, output_path="final_out")

        result_img = cv2.imread(result_path)
        return jsonify({"status": "success", "final_output": encode_image_to_base64(result_img)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Model-Based Floor Overlay ──────────────────────────────── #
@app.route("/overlayFloor", methods=["POST"])
def overlay_floor_model():
    try:
        data = request.json
        room_image_b64 = data.get("room_image")
        design_image_b64 = data.get("design_image")

        if not room_image_b64 or not design_image_b64:
            return jsonify({"error": "Both room_image and design_image must be provided"}), 400

        unique_id = str(uuid.uuid4())
        room_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        design_path = os.path.join("inputTile", f"design_{unique_id}.jpg")
        mask_path = os.path.join("mask_out", f"mask_{unique_id}.jpg")
        final_path = os.path.join("final_out", f"final_{unique_id}.jpg")

        room_img = decode_base64_to_image(room_image_b64)
        design_img = decode_base64_to_image(design_image_b64)
        cv2.imwrite(room_path, room_img)
        cv2.imwrite(design_path, design_img)

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
        return jsonify({"error": str(e)}), 500

# # ─── Computational Floor Overlay ────────────────────────────── #
# @app.route("/overlayFloorComputational", methods=["POST"])
# def overlay_floor_computational():
#     try:
#         data = request.json
#         room_image_b64 = data.get("room_image")
#         design_image_b64 = data.get("design_image")
#         height_mul = data.get("height_mul", 2)
#         width_mul = data.get("width_mul", 3)

#         if not room_image_b64 or not design_image_b64:
#             return jsonify({"error": "Both room_image and design_image must be provided"}), 400

#         unique_id = str(uuid.uuid4())
#         room_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
#         design_path = os.path.join("inputTile", f"design_{unique_id}.jpg")

#         room_img = decode_base64_to_image(room_image_b64)
#         design_img = decode_base64_to_image(design_image_b64)
#         cv2.imwrite(room_path, room_img)
#         cv2.imwrite(design_path, design_img)

#         final_path = overlay(room_path, design_path, height_mul, width_mul)
#         final_output = cv2.imread(final_path)

#         if final_path:
#             return jsonify({"status": "success", "final_output": encode_image_to_base64(final_output)})
#         else:
#             return jsonify({"error": "Failed to generate final output"}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    app.run(debug=True)
