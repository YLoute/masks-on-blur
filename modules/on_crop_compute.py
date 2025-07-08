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
import math

def meanPointPerValueNorm(im, threshold=120):
    im = cv2.GaussianBlur(im, (9, 9), 0)
    im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, im2 = cv2.threshold(im, threshold, 255, cv2.THRESH_BINARY)
    return im2

def canny(im):
    #blurred = cv2.GaussianBlur(im, (5, 5), 0)
    im = cv2.Canny(im, 80, 200, L2gradient = False)
    return im

def canny_lines(im):
    white_pixels = np.where(im > 0)
    height, width = im.shape
    visited = np.zeros_like(im, dtype=bool)
    lines = []
    size = 2 # Size for researching pixels from same line
    def get_neighbors(y, x):
        for dy in range(-1-size, size):
            for dx in range(1-size, size):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    yield ny, nx

    for y, x in zip(*white_pixels):
        if visited[y, x]:
            continue

        line = [(x, y)]
        visited[y, x] = True
        stack = [(y, x)]

        while stack:
            cy, cx = stack.pop()
            for ny, nx in get_neighbors(cy, cx):
                if im[ny, nx] > 0 and not visited[ny, nx]:
                    line.append((nx, ny))
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        if len(line) > 1:
            lines.append(np.array(line, dtype=np.int32))
    # Debug if not enough points detected
    #print("Number of Canny lines " +str(len(lines))+" in cropzone size ("+str(width)+", "+str(height)+")")
    return lines

def canny_contours_to_center(lines):
    if len(lines) >= 2:
        sorted_contours = sorted(lines, key=lambda c: cv2.arcLength(c, False), reverse=True)
        longest_contours = sorted_contours[:2]
        return find_center(longest_contours[0], longest_contours[1])
    else:
        return False
    return coords

"""def find_center_old(contour1, contour2):
    min_dist = float('inf')
    nearest_points = None
    for point1 in contour1:
        for point2 in contour2:
            dist = np.linalg.norm(point1 - point2)
            if dist < min_dist:
                min_dist = dist
                nearest_points = (point1, point2)
    midpoint = ((nearest_points[0][0] + nearest_points[1][0]) // 2,
                (nearest_points[0][1] + nearest_points[1][1]) // 2)
    return midpoint"""

def find_center(contour1, contour2):
    contour1 = np.array(contour1).reshape(-1, 2)  # Transform as array with auto lines number (-1) and 2-layers: (x,y)
    contour2 = np.array(contour2).reshape(-1, 2)
    distances = np.linalg.norm(contour1[:, None, :] - contour2[None, :, :], axis=2) # (contour1, contour2, 2)
    min_dist_index = np.unravel_index(np.argmin(distances), distances.shape)
    min_dist = distances[min_dist_index]
    
    point1 = contour1[min_dist_index[0]]
    point2 = contour2[min_dist_index[1]]
    # /!\ Returning coordinates of the pixel, not the center of it.
    midpixel = ((point1[0] + point2[0]) / 2,
                (point1[1] + point2[1]) / 2)
    return midpixel

def houghlines(im):
    cropzone_hough_img = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
    cropzone_hough_img = cv2.cvtColor(cropzone_hough_img, cv2.COLOR_GRAY2BGR)
    img_nb = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    canny_img = canny(img_nb)
    
    linesP = None
    #linesP = cv2.HoughLinesP(canny_img, rho=1, theta=np.pi/180, threshold=10, minLineLength=30, maxLineGap=10)
    
    lines = None
    #lines = cv2.HoughLines(canny_img,rho=1,theta=np.pi/180,threshold=80)
    lines = cv2.HoughLines(canny_img,rho=1,theta=np.pi/180,threshold=50)
    
    if lines is not None:
        for line in lines:
            #print(line)
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(cropzone_hough_img, (x1, y1), (x2, y2), (255, 255, 0), 1)
    
    if linesP is not None:
        print(linesP)
        for lineP in linesP:
            #print(lineP)
            cv2.line(cropzone_hough_img, (lineP[0][0], lineP[0][1]), (lineP[0][2], lineP[0][3]), (0, 0, 255), 1)
    
    return cropzone_hough_img
    """
    #lines = cv2.HoughLines(im,rho=1,theta=np.pi/180,threshold=130)
    im2 = im.copy()
    for rho,theta in lines[0]:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(im2,(x1,y1),(x2,y2),(0,0,255),2)"""
    return im2
