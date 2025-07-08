######
# Created to generate masks with Laplacian Pyramid method on blurred areas for close-range Structure from Motion (SfM)
# Bonus: refine detection of AprilTags centers and write it in csv files
#
# Author: Yannick FAURE
# Licence: GPL v3
#
# Caution : This python code is not optimized and ressembles as a draft, however it works as intended.
######
# This code is the toolbox for Difference Of Gaussian (Laplacian Pyramids) and for morphological operations called by MAIN file
######

import cv2
import numpy as np

def create_Laplacian_Pyramid(img, num_levels):
    gaussian_pyramid = [img.astype(np.float32)] #more precise convolution operations (float64 irrelevant)
    for _ in range(num_levels):
        img = cv2.pyrDown(img) # Reduce /2/2 size of the img resulting as a Gaussian blur type kernel size = 5
        """
                Testing something different than OpenCV gauss convolution for downsizing)
        """
        #tmp_img = gaussian_convolution(tmp_img, gaussian_kernel_size) # Convolve with former type of kernel, size adapable
        #tmp_img = convolution_through_kernel(tmp_img) # Convolve with former type of kernel
        #tmp_img = wavelets(tmp_img) # Testing Wavelets
        gaussian_pyramid.append(img.astype(np.float32))
    
    laplacian_pyramid = []
    for i in range(num_levels):
        size = gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian_pyramid.append(np.subtract(gaussian_pyramid[i], gaussian_expanded))
    return laplacian_pyramid

def merge_pyramid(laplacian_pyramid):
    laplacian_merged = laplacian_pyramid[-1]
    for laplacian_img in reversed(laplacian_pyramid[:-1]):
        size = (laplacian_img.shape[1], laplacian_img.shape[0])
        laplacian_merged = cv2.pyrUp(laplacian_merged, dstsize=size)
        laplacian_merged = cv2.add(laplacian_merged, laplacian_img)
        # NB: OpenCV image addition: saturation addition capped at 255 if uint8. For instance 250+10 => 255 interest of clipping for float32, not for uint8.
    laplacian_merged = np.clip(laplacian_merged, 0, 255).astype(np.uint8)
    return laplacian_merged
    
    
def dilate_white_zones(img, value):
    action_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (value, value))
    img_dilate = cv2.dilate(img, action_kernel)
    #img_dilate = cv2.morphologyEx(img, cv2.MORPH_OPEN, action_kernel)
    return img_dilate

def erode_white_zones(img, value):
    if value < 1:
        return img
    if value % 2 == 0:
        value = value + 1
    action_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (value, value))
    img_erode = cv2.erode(img, action_kernel, iterations=1)
    return img_erode