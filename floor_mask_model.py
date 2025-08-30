# 001

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from transformers import AutoImageProcessor, MaskFormerModel
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from numba import njit, prange

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

feature_extractor = None
model = None
device = torch.device('cpu')

@njit(parallel=True)
def create_wall_overlay(mask,dsgn,woverlay):
    """
    Overlays a design pattern onto a mask, creating a new image with the design.

    Args:
        mask (np.array): The segmentation mask.
        dsgn (np.array): The design pattern image.
        woverlay (np.array): The output image to be created.

    Returns:
        np.array: The image with the design overlaid.
    """
    w,h,_ = woverlay.shape
    dw,dh,_ = dsgn.shape
    for i in prange(0,w):
        for j in prange(0,h):
            if(mask[i][j][0] == 255):
                p = dsgn[i%dw][j%dh]
                woverlay[i][j]= p
    return woverlay

@njit(parallel=True)
def create_output_image(imagearray,walloverlayarray):
    """
    Merges a wall overlay onto the original image.

    Args:
        imagearray (np.array): The original image array.
        walloverlayarray (np.array): The overlay image array.

    Returns:
        np.array: The combined image with the overlay.
    """
    h,w,_ =  walloverlayarray.shape
    for i in prange(0,h):
        for j in prange(0,w):
            if(walloverlayarray[i][j].sum() > 0 ):
                imagearray[i][j] =  walloverlayarray[i][j]
    return imagearray.astype(np.uint8)

@njit(parallel=True)
def create_image_with_shadow(img_gray,hsv_image,walloverlayarray):
    """
    Adds a shadow effect to the wall overlay based on the original image's grayscale values.

    Args:
        img_gray (np.array): The grayscale version of the original image.
        hsv_image (np.array): The image converted to HSV color space.
        walloverlayarray (np.array): The wall overlay image.

    Returns:
        np.array: The HSV image with shadows applied.
    """
    h,w,_ =  hsv_image.shape
    hsvmin = np.min(hsv_image[:,:,2])
    hsvmax = np.max(hsv_image[:,:,2])
    for i in prange(0,h):
        for j in prange(0,w):
            if(walloverlayarray[i][j].sum() > 0 ):
                hsv_image[i][j][2] = abs(hsv_image[i][j][2] - (((img_gray[i][j]/1)-hsvmin)/(hsvmax-hsvmin))*100)
    print('|INFO| The shape of the HSV image is:', hsv_image.shape)
    return hsv_image.astype(np.uint8)

def load_model():
    """
    Initializes and loads the MaskFormer model and feature extractor.

    Returns:
        None
    """
    global feature_extractor,model,device
    # load MaskFormer fine-tuned on COCO panoptic segmentation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-ade")
    model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
    # model.to(device)
    print("|INFO| Model Successfully Loaded")
    # image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
    # model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")

def infer(imagepath,designimgpath,outputpath,mode = 3):
    """
    Performs inference on an image to generate a segmentation mask for a specified feature.

    Args:
        imagepath (str): The path to the input image.
        designimgpath (str): The path to the design image (unused in this function).
        outputpath (str): The path to save the output mask image.
        mode (int, optional): The label ID for the feature to segment. Defaults to 3 (floors).

    Returns:
        int: 1 if inference is successful, 0 otherwise.
    """
    #mode 0 for walls
    #model 3 for floors
    #model 28 for carpet
    global feature_extractor,model,device
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(device)
    image = Image.open(imagepath).convert('RGB')
    inputs = feature_extractor(images=image, return_tensors="pt")
    # inputs = feature_extractor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    # class_queries_logits = outputs.class_queries_logits
    # masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to feature_extractor for postprocessing
    result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_panoptic_map = result["segmentation"].cpu()

    # Checking if the requested feature is in the image 
    if (mode not in [info['label_id'] for info in result['segments_info']]):
        print(f"|WARNING| The requested feature with mode {mode} was not found in the image.")
        return 0

    # Finding the id of the wall from the segment predictions
    # facebook/maskformer-swin-base-coco" -> 131
    # facebook/maskformer-swin-base-ade => 0
    wallitem = next(item for item in result['segments_info'] if item["label_id"] == mode)
    wallitemid = wallitem['id']

    #creating empty panoptic map
    color_predicted_panoptic_map = np.zeros((predicted_panoptic_map.shape[0], predicted_panoptic_map.shape[1], 3), dtype=np.uint8) # height, width, 3
    color_predicted_panoptic_map[predicted_panoptic_map == wallitemid ] = (255,0,0)

    plt.imsave(outputpath,color_predicted_panoptic_map)
    print('|INFO| Inference done!')
    if torch.cuda.is_available():
        model.to('cpu')
        del inputs,outputs,result
        torch.cuda.empty_cache()
    print('|INFO| CUDA memory allocated:', torch.cuda.memory_allocated())
    return 1

def main():
    """
    Main function to run the floor mask generation process.
    It loads the model and performs inference on a sample image.
    """
    try:
        image_path = "../Floor-Overlay/inputRoom/room_01a80ef6-b94e-4f2d-8169-84f1b0ec3896.jpg"
        design_img_path = None # This parameter is not used in this specific file, so it's set to None for clarity.
        output_path = "../Floor-Overlay/mask_out/mask_01a80ef6-b94e-4f2d-8169-84f1b0ec3896.jpg"
        
        load_model()
        success = infer(image_path, design_img_path, output_path, mode=3)
        
        if success:
            print(f"|OUTPUT| Floor mask successfully saved to: {output_path}")
        else:
            print(f"|ERROR| Failed to generate a floor mask for the image: {image_path}")
    except Exception as e:
        print(f"|ERROR| An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()