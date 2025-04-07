# 017

import os
import cv2
import base64
import uuid
import numpy as np
from flask import Flask, request, jsonify
from mask_room_image import mask
from scale_and_overlay import place_on_black, create_black_image
from convert_binary import convert_to_binary_mask, convert_to_binary_carpet
from carpet_circle import carpet_ellipse_and_center
from find_centroid import find_and_mark_floor_center

app = Flask(__name__)

# Ensure directories exist
os.makedirs("inputRoom", exist_ok=True)
os.makedirs("inputCarpet", exist_ok=True)
os.makedirs("temporary", exist_ok=True)
os.makedirs("final_out", exist_ok=True)

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

@app.route("/overlay_carpet", methods=["POST"])
def overlay_carpet_api():
    try:
        data = request.json
        room_image_b64 = data.get("room_image")
        carpet_image_b64 = data.get("carpet_image")

        if not room_image_b64 or not carpet_image_b64:
            return jsonify({"error": "Both room_image and carpet_image must be provided"}), 400

        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        room_img_path = os.path.join("inputRoom", f"room_{unique_id}.jpg")
        carpet_img_path = os.path.join("inputCarpet", f"carpet_{unique_id}.png")  # allow alpha

        # Decode and save
        room_img = decode_base64_to_image(room_image_b64)
        carpet_img = decode_base64_to_image(carpet_image_b64)

        cv2.imwrite(room_img_path, room_img)
        cv2.imwrite(carpet_img_path, carpet_img)

        # --- Begin overlay logic ---
        room_image_name = os.path.splitext(os.path.basename(room_img_path))[0]
        floor_center = find_and_mark_floor_center(room_img_path)
        if not floor_center:
            return jsonify({"error": "Floor center not found"}), 400

        fx, fy = floor_center
        black_bg_path = create_black_image(room_img_path)
        black_bg = cv2.imread(black_bg_path)
        carpet_path, carpet_center = carpet_ellipse_and_center(carpet_img_path)
        carpet = cv2.imread(carpet_path, cv2.IMREAD_UNCHANGED)
        cx, cy = carpet_center

        top_left_x = fx - cx
        top_left_y = fy - cy
        h, w = carpet.shape[:2]
        overlay = black_bg.copy()

        x1 = max(top_left_x, 0)
        y1 = max(top_left_y, 0)
        x2 = min(top_left_x + w, overlay.shape[1])
        y2 = min(top_left_y + h, overlay.shape[0])

        crop_x1 = x1 - top_left_x
        crop_y1 = y1 - top_left_y
        crop_x2 = crop_x1 + (x2 - x1)
        crop_y2 = crop_y1 + (y2 - y1)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return jsonify({"error": "Invalid crop dimensions for overlay"}), 500

        roi = overlay[y1:y2, x1:x2]
        carpet_crop = carpet[crop_y1:crop_y2, crop_x1:crop_x2]

        if carpet_crop.shape[2] == 4:
            b, g, r, a = cv2.split(carpet_crop)
            carpet_rgb = cv2.merge((b, g, r))
            alpha_mask = cv2.merge((a, a, a)) / 255.0
        else:
            carpet_rgb = carpet_crop
            alpha_mask = np.ones_like(carpet_rgb, dtype=np.float32)

        blended = (roi * (1 - alpha_mask) + carpet_rgb * alpha_mask).astype(np.uint8)
        overlay[y1:y2, x1:x2] = blended

        room_img_loaded = cv2.imread(room_img_path)
        final_output = cv2.addWeighted(room_img_loaded, 1.0, overlay, 1.0, 0)

        final_output_path = os.path.join("final_out", f"final_overlay_{unique_id}.jpg")
        cv2.imwrite(final_output_path, final_output)
        final_output_b64 = encode_image_to_base64(final_output)

        return jsonify({"status": "success", "final_output": final_output_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
