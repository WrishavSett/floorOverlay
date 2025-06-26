import os
import base64
import requests
from itertools import product

# Base URL
BASE_URL = "http://127.0.0.1:5000"

# Paths to your image folders
ROOMS_DIR = "sample_images/rooms"
DESIGNS_DIR = "sample_images/designs"
CARPETS_DIR = "sample_images/carpets"

# Output directory
OUTPUT_DIR = "batch_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility Functions
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def save_base64_image(base64_string, filename):
    image_data = base64.b64decode(base64_string)
    with open(filename, "wb") as file:
        file.write(image_data)

def send_overlay_carpet(room_b64, carpet_b64, overlay_type="ellipse"):
    payload = {
        "room_image": room_b64,
        "carpet_image": carpet_b64,
        "overlay_type": overlay_type
    }
    response = requests.post(f"{BASE_URL}/overlayCarpet", json=payload)
    return response

def send_overlay_floor_model(room_b64, design_b64):
    payload = {
        "room_image": room_b64,
        "design_image": design_b64
    }
    response = requests.post(f"{BASE_URL}/overlayFloor", json=payload)
    return response

def send_overlay_floor_computational(room_b64, design_b64, height_mul=2, width_mul=3):
    payload = {
        "room_image": room_b64,
        "design_image": design_b64,
        "height_mul": height_mul,
        "width_mul": width_mul
    }
    response = requests.post(f"{BASE_URL}/overlayFloorComputational", json=payload)
    return response

# ───────────────────────────────────────────────────────────── #

def batch_process():
    rooms = [os.path.join(ROOMS_DIR, f) for f in os.listdir(ROOMS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    designs = [os.path.join(DESIGNS_DIR, f) for f in os.listdir(DESIGNS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    carpets = [os.path.join(CARPETS_DIR, f) for f in os.listdir(CARPETS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Found {len(rooms)} room images, {len(designs)} design images, {len(carpets)} carpet images.")

    # # Code to process all combinations of rooms, designs, and carpets
    # for room_path, design_path, carpet_path in product(rooms, designs, carpets):
    #     room_name = os.path.splitext(os.path.basename(room_path))[0]
    #     design_name = os.path.splitext(os.path.basename(design_path))[0]
    #     carpet_name = os.path.splitext(os.path.basename(carpet_path))[0]

    #     print(f"\n🔵 Processing Room: {room_name}, Design: {design_name}, Carpet: {carpet_name}")

    #     # Read base64 encodings
    #     room_b64 = image_to_base64(room_path)
    #     design_b64 = image_to_base64(design_path)
    #     carpet_b64 = image_to_base64(carpet_path)

    #     # 1. Carpet Overlay (Ellipse)
    #     resp_carpet = send_overlay_carpet(room_b64, carpet_b64, overlay_type="ellipse")
    #     if resp_carpet.status_code == 200:
    #         save_path = os.path.join(OUTPUT_DIR, f"{room_name}_{carpet_name}_carpet_ellipse.jpg")
    #         save_base64_image(resp_carpet.json()["final_output"], save_path)
    #         print(f"✅ Saved: {save_path}")
    #     else:
    #         print(f"❌ Carpet overlay failed: {resp_carpet.json()}")

    #     # 2. Carpet Overlay (Trapezoid)
    #     resp_carpet_trap = send_overlay_carpet(room_b64, carpet_b64, overlay_type="trapezoid")
    #     if resp_carpet_trap.status_code == 200:
    #         save_path = os.path.join(OUTPUT_DIR, f"{room_name}_{carpet_name}_carpet_trapezoid.jpg")
    #         save_base64_image(resp_carpet_trap.json()["final_output"], save_path)
    #         print(f"✅ Saved: {save_path}")
    #     else:
    #         print(f"❌ Carpet trapezoid overlay failed: {resp_carpet_trap.json()}")

    #     # 3. Model-based Floor Overlay
    #     resp_floor_model = send_overlay_floor_model(room_b64, design_b64)
    #     if resp_floor_model.status_code == 200:
    #         save_path = os.path.join(OUTPUT_DIR, f"{room_name}_{design_name}_floor_model.jpg")
    #         save_base64_image(resp_floor_model.json()["final_output"], save_path)
    #         print(f"✅ Saved: {save_path}")
    #     else:
    #         print(f"❌ Floor model overlay failed: {resp_floor_model.json()}")

    #     # # 4. Computational Floor Overlay
    #     # resp_floor_comp = send_overlay_floor_computational(room_b64, design_b64)
    #     # if resp_floor_comp.status_code == 200:
    #     #     save_path = os.path.join(OUTPUT_DIR, f"{room_name}_{design_name}_floor_computational.jpg")
    #     #     save_base64_image(resp_floor_comp.json()["final_output"], save_path)
    #     #     print(f"✅ Saved: {save_path}")
    #     # else:
    #     #     print(f"❌ Floor computational overlay failed: {resp_floor_comp.json()}")

    # Process all combinations of rooms and carpets
    for room_path, design_path, in product(rooms, designs):
        room_name = os.path.splitext(os.path.basename(room_path))[0]
        design_name = os.path.splitext(os.path.basename(design_path))[0]

        print(f"\n🔵 Processing Room: {room_name}, Design: {design_name}")

        # Read base64 encodings
        room_b64 = image_to_base64(room_path)
        design_b64 = image_to_base64(design_path)

        # 3. Model-based Floor Overlay
        resp_floor_model = send_overlay_floor_model(room_b64, design_b64)
        if resp_floor_model.status_code == 200:
            save_path = os.path.join(OUTPUT_DIR, f"{room_name}_{design_name}_floor_model.jpg")
            save_base64_image(resp_floor_model.json()["final_output"], save_path)
            print(f"✅ Saved: {save_path}")
        else:
            print(f"❌ Floor model overlay failed: {resp_floor_model.json()}")

# ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    batch_process()