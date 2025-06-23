#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:06:18 2024

@author: sam
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
# Function to draw contours
def draw_contour(event, x, y, flags, param):
    global drawing, contour, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        contour = [(x, y)]
        img_copy = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            contour.append((x, y))
            cv2.line(img_copy, contour[-2], contour[-1], (255, 0, 0), 2)
            cv2.imshow('Image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        contour.append((x, y))
        cv2.line(img_copy, contour[-2], contour[-1], (255, 0, 0), 2)
        cv2.imshow('Image', img_copy)

def enhance_contrast(image):
    """
    Enhance contrast, especially revealing dim objects, 
    using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """

    # 1) Convert from BGR to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) Create a CLAHE object (feel free to tweak clipLimit and tileGridSize)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # 3) Apply CLAHE
    eq_gray = clahe.apply(gray)

    # 4) Convert back to BGR if you need color lines, etc.
    image_eq = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2BGR)

    return image_eq

# Load an image
image_path = '/home/sam/bucket/sam/igl_wave/'
img_name=glob.glob(image_path + '/*frame_0.png')

#img_name=glob.glob(image_path + '*01*dolp.png')
#img_name=glob.glob(image_path + '*01_rev_dolp.png')

img_name=['/home/sam/bucket/Gio/Polarization/pol-ccard_depth-dist/20250503-shimp-on-transect/raws/shrimp/extras/P5033002_green.png']

for im_name in img_name:

    img = cv2.imread(im_name)
    
    img = enhance_contrast(img)
    img_copy = img.copy()
    
    # Create a window and set a mouse callback
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)  # Allow window resizing
    
    cv2.setMouseCallback('Image', draw_contour)
    
    drawing = False
    contour = []
    
    while True:
        cv2.imshow('Image', img_copy)
        cv2.resizeWindow('Image',1500, 900)  # Set the initial size
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
    
    cv2.destroyAllWindows()
    
    # Create a mask image of the same size as the original image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Draw the contour on the mask
    if contour:
        cv2.drawContours(mask, [np.array(contour)], -1, (255), thickness=cv2.FILLED)
    
    # Save the mask image
    cv2.imwrite(image_path +'/mask_image.jpg', mask)
   # cv2.imwrite(im_name +'_fg_image_raw.png', mask)
    
    # Display the mask image using matplotlib
    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.title('Mask Image')
    # plt.show()
