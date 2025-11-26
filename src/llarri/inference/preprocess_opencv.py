import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    # Add preprocessing logic (binarization, deskewing, etc.)
    return img
