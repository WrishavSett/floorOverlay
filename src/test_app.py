import os
import base64
import requests
from itertools import product

from src.utils import *
from src.main import *

# Base URL of Flask server
BASE_URL = "http://127.0.0.1:5001"

# Input directories
ROOMS_DIR = "sample_images/rooms"
DESIGNS_DIR = "sample_images/designs"
CARPETS_DIR = "sample_images/carpets"

# Output directory
OUTPUT_DIR = "batch_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def send_overlay_carpet(
        room_b64,
        carpet_b64,
        overlay_type="ellipse"):
    """
    Sends a request to the server to overlay a carpet image on a room image.
    
    Args:
        room_b64 (str): Base64-encoded room image.
        carpet_b64 (str): Base64-encoded carpet image.
        overlay_type (str): The type of overlay (e.g., "ellipse").
        
    Returns:
        requests.Response: The response object from the server.
    """
    payload = {
        "room_image": room_b64,
        "carpet_image": carpet_b64,
        "overlay_type": overlay_type
    }
    return requests.post(f"{BASE_URL}/overlayCarpet", json=payload)

def send_overlay_floor_model(
        room_b64,
        design_b64):
    """
    Sends a request to the server to overlay a floor design on a room image using a floor segmentation model.
    
    Args:
        room_b64 (str): Base64-encoded room image.
        design_b64 (str): Base64-encoded floor design image.
        
    Returns:
        requests.Response: The response object from the server.
    """
    payload = {
        "room_image": room_b64,
        "design_image": design_b64
    }
    return requests.post(f"{BASE_URL}/overlayFloor", json=payload)

# ───── Batch Processing Logic ─────────────────────────────── #
def batch_process():
    """
    Iterates through all combinations of room, floor design, and carpet images,
    sends them to the server for processing, and saves the results.
    
    Returns:
        None
    """
    rooms = [os.path.join(ROOMS_DIR, f) for f in os.listdir(ROOMS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    designs = [os.path.join(DESIGNS_DIR, f) for f in os.listdir(DESIGNS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    carpets = [os.path.join(CARPETS_DIR, f) for f in os.listdir(CARPETS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"|INFO| Found {len(rooms)} room images, {len(designs)} design images, {len(carpets)} carpet images.")

    for room_path, design_path, carpet_path in product(rooms, designs, carpets):
        room_name = os.path.splitext(os.path.basename(room_path))[0]
        design_name = os.path.splitext(os.path.basename(design_path))[0]
        carpet_name = os.path.splitext(os.path.basename(carpet_path))[0]

        print(f"\n|INFO| Processing Room: {room_name}, Design: {design_name}, Carpet: {carpet_name}")

        try:
            # Encode images to base64
            room_b64 = image_to_base64(room_path)
            design_b64 = image_to_base64(design_path)
            carpet_b64 = image_to_base64(carpet_path)

            # ── Carpet Overlays ──
            for overlay_type in ["ellipse", "trapezoid"]:
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

# ───────────────────────────────────────────────────────────── #

def main():
    """
    Main function to run the batch processing of images.
    """
    batch_process()

if __name__ == "__main__":
    main()