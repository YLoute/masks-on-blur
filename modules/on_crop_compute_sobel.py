######
# Created to generate masks with Laplacian Pyramid method on blurred areas for close-range Structure from Motion (SfM)
# Bonus: refine detection of AprilTags centers and write it in csv files
#
# Author: Yannick FAURE
# Licence: GPL v3
#
# Caution : This python code is not optimized and ressembles as a draft, however it works as intended.
######
# This code is related to AprilTags 36h11 (WITH A CHECKERBOARD pattern at the center) refining center detection
# It includes various tests to improve center detection
######

import numpy as np
import cv2

def sobel(im):
    sobel_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude of gradient
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize magnitude for visualisation
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return magnitude
    
def sob(im):
    magnitude = sobel(im)
    blurred = cv2.GaussianBlur(magnitude, (9, 9), 0)
    normalized_image = cv2.normalize(blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_image

def monge(im, margin):
    """
    Monge s²-rt > 0 equals special point
    """
    r = cv2.Sobel(im, cv2.CV_64F, dx=2, dy=0, ksize=5)  # ∂²I/∂x²
    t = cv2.Sobel(im, cv2.CV_64F, dx=0, dy=2, ksize=5)  # ∂²I/∂y²
    s = cv2.Sobel(im, cv2.CV_64F, dx=1, dy=1, ksize=5)  # ∂²I/∂x∂y
    
    s_squared = np.square(s) # Calc s²
    
    monge = s_squared - r * t
    
    # Create mask to exclude pixels in margin
    mask = np.ones(monge.shape[:5], dtype=bool)
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    
    monge = monge * mask
    
    return monge

def monge_positive(im, num_points):
    sy, sx = im.shape
    gauss_kernel_size = 5
    blurred = cv2.GaussianBlur(im, (gauss_kernel_size, gauss_kernel_size), 0)
    #relative_margin = gauss_kernel_size/2+1
    print(sx)
    relative_margin = int((sx-1)/2)
    monge_matrix = monge(blurred, relative_margin)
    #positive_indices = np.where(monge_matrix > 0)
    positive_indices = np.unravel_index(np.argsort(monge_matrix, axis=None)[-num_points:][::-1], monge_matrix.shape)
    positive_points = [(x, y, monge_matrix[y, x]) for y, x in zip(*positive_indices)]
    
    return positive_points
    

def find_average_pixel(image, threshold=200):
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    coords = np.nonzero(thresholded)
    y_coords, x_coords = coords[0], coords[1]
    
    if len(x_coords) > 0 and len(y_coords) > 0:
        avg_x = np.mean(x_coords)
        avg_y = np.mean(y_coords)
        return (int(avg_x), int(avg_y))
    else:
        return None
    
"""def contours_to_center2(im, threshold=200):
    im = sob(im)
    _, im = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # Obtenir les indices des pixels blancs
    white_pixels = np.where(im)
    
    kernel = np.ones((9, 9), dtype=np.float32)
    neighbor_score = cv2.filter2D(im, -1, kernel)
    
    #for y, x in zip(*white_pixels):
    for px in white_pixels:
        if n"""
        
        
    
def contours_to_center(im):
    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width = im.shape[:2]
    processed_contours = []
    for contour in contours:
        filtered_contour = [point for point in contour if not (point[0][0] == 0 or point[0][0] == width-1 or
                                                               point[0][1] == 0 or point[0][1] == height-1)]
    
        # Check continuity
        if filtered_contour:
            continuous_contour = [filtered_contour[0]]
            for i in range(1, len(filtered_contour)):
                if np.all(np.abs(filtered_contour[i] - filtered_contour[i-1]) <= 10):
                    continuous_contour.append(filtered_contour[i])
                else:
                    if len(continuous_contour) > 1:
                        processed_contours.append(np.array(continuous_contour))
                    continuous_contour = [filtered_contour[i]]
            
            if len(continuous_contour) > 1:
                processed_contours.append(np.array(continuous_contour))
        
        if len(continuous_contour) > 1:
            processed_contours.append(np.array(continuous_contour))
    
    if len(processed_contours) >= 2:
        #print("contours")
        #print(contours)
        sorted_contours = sorted(processed_contours, key=lambda c: cv2.arcLength(c, False), reverse=True)
        longest_contours = sorted_contours[:2]
        return find_nearest_points(longest_contours[0], longest_contours[1])
    else:
        return False
    
    
    
def find_nearest_points(contour1, contour2):
    min_dist = float('inf')
    nearest_points = None
    for point1 in contour1:
        for point2 in contour2:
            dist = np.linalg.norm(point1 - point2)
            if dist < min_dist:
                min_dist = dist
                nearest_points = (point1[0], point2[0])
    midpoint = ((nearest_points[0][0] + nearest_points[1][0]) // 2,
                (nearest_points[0][1] + nearest_points[1][1]) // 2)
    return midpoint

    
    
def weighted_center(image):
    processed = sob(image)
    
    # Square to weight more differences
    weighted = processed.astype(np.float64) #** 2
    
    # Grids of coordinates
    indices = np.indices(weighted.shape)
    y, x = indices[0], indices[1]
    
    # Sum weights
    total_weight = np.sum(weighted)
    
    # Avoid dividing by 0
    if total_weight == 0:
        return None
    
    # Compute weighted mean coordinates
    center_x = np.sum(x * weighted) / total_weight
    center_y = np.sum(y * weighted) / total_weight
    
    return (int(center_x), int(center_y))

