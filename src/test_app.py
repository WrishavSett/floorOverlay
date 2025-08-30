import os
import base64
import requests
from itertools import product
from utils import image_to_base64, save_base64_image

# Base URL of Flask server
BASE_URL = "http://127.0.0.1:5001"

# Input directories
ROOMS_DIR = "sample_images/rooms"
DESIGNS_DIR = "sample_images/designs"
CARPETS_DIR = "sample_images/carpets"

# Output directory
OUTPUT_DIR = "batch_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───── Helper Functions ───────────────────────────────────── #

def send_overlay_floor_model(room_b64, design_b64):
    """
    Sends a POST request to the floor overlay endpoint.
    """
    url = f"{BASE_URL}/overlay_floor_model"
    headers = {"Content-Type": "application/json"}
    payload = {
        "room_image": room_b64,
        "design_image": design_b64
    }
    print("|INFO| Sending request to /overlay_floor_model...")
    return requests.post(url, headers=headers, json=payload)

def send_overlay_carpet(room_b64, carpet_b64, overlay_type):
    """
    Sends a POST request to the carpet overlay endpoint.
    """
    url = f"{BASE_URL}/overlay_carpet_trapezoid"
    headers = {"Content-Type": "application/json"}
    payload = {
        "room_image": room_b64,
        "carpet_image": carpet_b64,
        "overlay_type": overlay_type
    }
    print(f"|INFO| Sending request to /overlay_carpet_trapezoid for type: {overlay_type}...")
    return requests.post(url, headers=headers, json=payload)

# ───── Main Test Function ───────────────────────────────────── #
def batch_process():
    """
    Tests the endpoints with a batch of images.
    """
    room_files = [f for f in os.listdir(ROOMS_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]
    design_files = [f for f in os.listdir(DESIGNS_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]
    carpet_files = [f for f in os.listdir(CARPETS_DIR) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    if not all([room_files, design_files, carpet_files]):
        print("|ERROR| Ensure all input directories have images.")
        return

    combinations = list(product(room_files, design_files, carpet_files))
    print(f"|INFO| Processing {len(combinations)} combinations...")

    for room_name, design_name, carpet_name in combinations:
        try:
            print(f"|INFO| Processing combination: {room_name}, {design_name}, {carpet_name}")
            room_path = os.path.join(ROOMS_DIR, room_name)
            design_path = os.path.join(DESIGNS_DIR, design_name)
            carpet_path = os.path.join(CARPETS_DIR, carpet_name)

            room_b64 = image_to_base64(room_path)
            design_b64 = image_to_base64(design_path)
            carpet_b64 = image_to_base64(carpet_path)
            
            # ── Generic Carpet Overlay Test ──
            for overlay_type in ["trapezoid", "ellipse", "transparent"]:
                resp = send_overlay_carpet(room_b64, carpet_b64, overlay_type=overlay_type)
                if resp.status_code == 200:
                    result = resp.json()
                    save_path = os.path.join(OUTPUT_DIR, f"{room_name}_{carpet_name}_carpet_{overlay_type}.png")
                    save_base64_image(result["transparent_carpet_image"], save_path)
                    print(f"|OUTPUT| Saved: {save_path}")
                else:
                    print(f"|WARNING| Carpet overlay ({overlay_type}) failed: {resp.json()}")

            # ── Model-Based Floor Overlay ──
            resp_floor = send_overlay_floor_model(room_b64, design_b64)
            if resp_floor.status_code == 200:
                result = resp_floor.json()
                save_path = os.path.join(OUTPUT_DIR, f"{room_name}_{design_name}_floor_model.jpg")
                save_base64_image(result["final_output"], save_path)
                print(f"|OUTPUT| Saved: {save_path}")
            else:
                print(f"|WARNING| Floor model overlay failed: {resp_floor.json()}")

        except Exception as e:
            print(f"|ERROR| Exception during batch process for combination {room_name}, {design_name}, {carpet_name}: {str(e)}")

if __name__ == "__main__":
    batch_process()