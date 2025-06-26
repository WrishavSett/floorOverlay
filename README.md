
# Carpet and Floor Overlay System

This repository provides a complete system for overlaying carpet and floor textures on room images using deep learning and geometric transformation techniques. It is designed to be used via a Flask-based REST API and includes utilities for preprocessing, inference, transformation, and batch testing.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Requirements](#requirements)  
5. [Installation](#installation)  
6. [Module Descriptions](#module-descriptions)  
7. [Running the Flask API](#running-the-flask-api)  
8. [API Endpoints](#api-endpoints)  
9. [Testing with Postman](#testing-with-postman)  
10. [Batch Testing](#batch-testing)  
11. [Outputs](#outputs)

---

## Overview

This system is designed to provide realistic overlays of carpets and floor textures on indoor room images. The goal is to simulate interior decoration effects using advanced image processing techniques and deep learning models. The application supports API-based access for seamless integration into design workflows.

---

## Features

- Carpet overlays using:
  - Elliptical transformation
  - Trapezoidal warping via homography

- Floor overlays using:
  - Deep learning segmentation (MaskFormer from HuggingFace)
  - Computational geometric warping

- Robust batch testing across combinations of rooms, carpets, and floor designs

- Clean Flask-based API access for automated integration

- Modular and extendable codebase

---

## Project Structure

```plaintext
.
├── app.py                         # Main Flask API server
├── test_app.py                    # Batch testing utility
├── carpet_circle.py               # Elliptical carpet warping logic
├── carpet_working.py              # Trapezoidal carpet overlay using contours and homography
├── convert_binary.py              # Generates binary masks from images
├── find_centroid.py               # Locates centroid of the floor region in a mask
├── floor_mask_model.py           # Loads and runs MaskFormer for floor segmentation
├── floor_overlay.py               # Full computational floor overlay using perspective warping
├── mask_room_image.py             # Interface to run floor segmentation and save the mask
├── overlay.py                     # Logic to combine carpet/floor overlays with room image
├── scale_and_overlay.py           # Carpet placement and resizing logic
├── sample_images/
│   ├── carpets/
│   ├── designs/
│   └── rooms/
├── batch_outputs/                 # Outputs from batch testing
├── final_out/                     # Outputs from single API runs
└── requirements.txt               # Dependencies
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Required Libraries

- Flask: Web API
- OpenCV: Image processing
- NumPy: Array operations
- Torch: Deep learning
- Transformers: HuggingFace pretrained models
- Pillow: Image manipulation
- Matplotlib: Visualization
- Numba: Speed-up routines
- Requests, Pandas: API and batch tools

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/carpet-overlay-system.git
cd carpet-overlay-system
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Place your sample images under:

```plaintext
sample_images/
├── carpets/
├── designs/
└── rooms/
```

---

## Module Descriptions

### `app.py`

- Hosts a Flask server with three endpoints:
  - `/overlayCarpet`: Places a carpet image on a room floor.
  - `/overlayFloor`: Uses semantic segmentation to extract floor mask and apply design.
  - `/overlayFloorComputational`: Uses geometric warping to apply floor designs.
- Handles decoding of base64 input images and encoding of output.

---

### `test_app.py`

- Automates the process of testing all image combinations.
- Uses Python's requests module to call APIs.
- Saves output in `batch_outputs/`.

---

### `carpet_circle.py`

- Converts rectangular carpets into ellipses.
- Uses geometric scaling to simulate 3D perspective.
- Outputs a warped elliptical carpet image on a black background.

---

### `scale_and_overlay.py`

- Receives the carpet and room mask.
- Rescales the carpet based on floor region size.
- Aligns the carpet center with detected floor centroid.

---

### `overlay.py`

- Coordinates different transformations and overlays:
  - Calls binary converters, resizers, warpers.
  - Applies binary masks and blends carpet with room.
- Supports both ellipse and trapezoid carpet modes.

---

### `carpet_working.py`

- Applies homography using room floor mask contours.
- Warps carpet image into a trapezoidal shape.
- Provides realism by simulating room angle.

---

### `convert_binary.py`

- Uses OpenCV to threshold and binarize carpet and room images.
- Masks are used to separate floor/carpet regions from the rest.

---

### `find_centroid.py`

- Identifies the centroid of red pixels in binary masks.
- Helps in accurate placement of carpets on floor area.

---

### `floor_mask_model.py`

- Loads pretrained MaskFormer model from HuggingFace.
- Performs semantic segmentation of floor region.
- Returns binary floor mask for room image.

---

### `mask_room_image.py`

- Wrapper around `floor_mask_model.py`.
- Used to store and reuse generated floor masks.

---

### `floor_overlay.py`

- Full computational pipeline:
  - Tiles the design image.
  - Applies a perspective transform.
  - Masks to floor area and blends.
- Simulates physical floor texture mapping without a DL model.

---

## Running the Flask API

```bash
python app.py
```

Server runs on:

```
http://127.0.0.1:5000
```

---

## API Endpoints

### 1. `/overlayCarpet`

**Method**: `POST`

```json
{
  "room_image": "BASE64_ENCODED_ROOM",
  "carpet_image": "BASE64_ENCODED_CARPET",
  "overlay_type": "ellipse"  // or "trapezoid"
}
```

### 2. `/overlayFloor`

**Method**: `POST`

```json
{
  "room_image": "BASE64_ENCODED_ROOM",
  "design_image": "BASE64_ENCODED_FLOOR"
}
```

### 3. `/overlayFloorComputational`

**Method**: `POST`

```json
{
  "room_image": "BASE64_ENCODED_ROOM",
  "design_image": "BASE64_ENCODED_FLOOR",
  "height_mul": 2,
  "width_mul": 3
}
```

**Response for all**: Base64-encoded output image:

```json
{
  "final_output": "BASE64_ENCODED_OUTPUT_IMAGE"
}
```

---

## Testing with Postman

1. Start server:

```bash
python app.py
```

2. Convert images to base64:

```python
import base64
with open("image.jpg", "rb") as f:
    print(base64.b64encode(f.read()).decode("utf-8"))
```

3. In Postman, use:
- Method: `POST`
- URL: e.g. `http://127.0.0.1:5000/overlayCarpet`
- Headers: `Content-Type: application/json`
- Body: raw → JSON

---

## Batch Testing

```bash
python test_app.py
```

- Loads all combinations of carpets, designs, rooms
- Sends requests to appropriate endpoints
- Outputs saved in `batch_outputs/`

---

## Outputs

- API outputs: `final_out/`
- Batch outputs: `batch_outputs/`

---

## License

MIT License