
# Carpet and Floor Overlay System

This project provides a robust pipeline for applying carpets and floor textures onto room images using both deep learning and geometric transformation techniques. It includes a Flask-based API server and a suite of preprocessing, segmentation, transformation, and testing scripts.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Modules Description](#modules-description)
5. [How to Run](#how-to-run)
6. [Testing](#testing)
7. [Testing with Postman](#testing-with-postman)
8. [Output](#output)

---

## Features

- Carpet overlay using ellipse and trapezoidal warping
- Floor texture mapping using:
  - Deep learning (semantic segmentation with MaskFormer)
  - Geometric computation (perspective transformations)
- Batch testing support for multiple rooms, designs, and carpets
- Flask REST API for external integration
- Robust image preprocessing and transformation utilities

---

## Project Structure

```
.
├── app.py
├── test_app.py
├── carpet_circle.py
├── carpet_working.py
├── convert_binary.py
├── find_centroid.py
├── floor_mask_model.py
├── floor_overlay.py
├── mask_room_image.py
├── overlay.py
├── scale_and_overlay.py
├── sample_images/
│   ├── rooms/
│   ├── carpets/
│   └── designs/
└── batch_outputs/
```

---

## Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

**Required Packages**:
- Flask
- OpenCV
- NumPy
- Torch
- Transformers (HuggingFace)
- Pillow
- Pandas
- Requests
- Matplotlib
- Numba

---

## Modules Description

### 1. `app.py`
Flask-based server exposing three main endpoints:
- `/overlayCarpet` for applying a carpet (ellipse or trapezoid)
- `/overlayFloor` using MaskFormer model
- `/overlayFloorComputational` using geometric transforms

### 2. `floor_mask_model.py`
- Loads and runs inference with `MaskFormerForInstanceSegmentation` from HuggingFace.
- Generates segmentation masks identifying floor regions.

### 3. `mask_room_image.py`
- Utility that calls `floor_mask_model.infer()` and returns the saved mask path.

### 4. `convert_binary.py`
- Converts room and carpet images into binary masks.
- Useful for masking and boolean operations during overlay.

### 5. `find_centroid.py`
- Locates the center of the detected floor region (assumed red mask) for placing carpets accurately.

### 6. `carpet_circle.py`
- Crops carpets into circular shapes and stretches them into ellipse form to simulate 3D perspective.

### 7. `scale_and_overlay.py`
- Handles carpet resizing and placement.
- Places carpet on a black background aligned to the detected floor center.

### 8. `overlay.py`
- High-level logic to:
  - Warp carpets into trapezoids or ellipses
  - Apply binary mask-based overlay with the room image

### 9. `carpet_working.py`
- Uses mask and contour detection to warp tile textures using homography.
- Aligns tile textures to detected floor regions.

### 10. `floor_overlay.py`
- Performs the full pipeline for geometric floor texture overlay.
- Steps:
  - Base image creation
  - Tile image tiling
  - Perspective transform
  - Masking and cropping
  - Final image compositing

### 11. `test_app.py`
- Batch testing script.
- Sends base64-encoded images to API endpoints.
- Supports combinations of room, carpet, and floor design.
- Saves all results in the `batch_outputs/` directory.

---

## How to Run

### Step 1: Start the Flask Server

```bash
python app.py
```

### Step 2: Send Requests

You can use:
- Postman
- `test_app.py` for batch automation

### Step 3: Run the Tester

```bash
python test_app.py
```

---

## Testing with Postman

You can use [Postman](https://www.postman.com/) to manually test the API endpoints exposed by the Flask server.

### 1. Start the Flask Server

Run the server locally with:

```bash
python app.py
```

Ensure it's running at `http://127.0.0.1:5000`.

---

### 2. Prepare Base64-Encoded Inputs

Use an online tool like [https://www.base64-image.de/](https://www.base64-image.de/) or a Python script:

```python
import base64

with open("your_image.jpg", "rb") as img_file:
    encoded = base64.b64encode(img_file.read()).decode('utf-8')
    print(encoded)
```

---

### 3. Set Up Requests in Postman

#### a. Carpet Overlay

- **Endpoint**: `POST /overlayCarpet`
- **URL**: `http://127.0.0.1:5000/overlayCarpet`
- **Headers**:  
  `Content-Type: application/json`
- **Body (raw, JSON)**:
```json
{
  "room_image": "BASE64_ROOM_IMAGE_STRING",
  "carpet_image": "BASE64_CARPET_IMAGE_STRING",
  "overlay_type": "ellipse"
}
```

---

#### b. Model-Based Floor Overlay

- **Endpoint**: `POST /overlayFloor`
- **URL**: `http://127.0.0.1:5000/overlayFloor`
- **Body**:
```json
{
  "room_image": "BASE64_ROOM_IMAGE_STRING",
  "design_image": "BASE64_DESIGN_IMAGE_STRING"
}
```

---

#### c. Computational Floor Overlay

- **Endpoint**: `POST /overlayFloorComputational`
- **URL**: `http://127.0.0.1:5000/overlayFloorComputational`
- **Body**:
```json
{
  "room_image": "BASE64_ROOM_IMAGE_STRING",
  "design_image": "BASE64_DESIGN_IMAGE_STRING",
  "height_mul": 2,
  "width_mul": 3
}
```

---

### 4. Visualize the Output

- The response will contain a JSON object with a `final_output` field.
- This value is a base64 string of the resulting image.
- You can save it using a decoder like:

```python
import base64

with open("output.jpg", "wb") as f:
    f.write(base64.b64decode(FINAL_OUTPUT_STRING))
```

---

## Output

- Final images with overlays are stored in:
  - `final_out/` (when using API)
  - `batch_outputs/` (when using `test_app.py`)

---
