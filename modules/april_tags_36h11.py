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

import cv2
import numpy as np
import time
from pathlib import Path

import robotpy_apriltag as april
import modules.on_crop_compute as on_crop_compute
import modules.on_crop_compute_sobel as on_crop_compute_sobel

"""
def run_on_folder(folderpath):
    folder = Path(folderpath)
    for jpg_file in folder.glob("*.[jJ][pP][gG]"):
        #global start_time
        #if start_time == 0:
        #    start_time = time.time()
        print(jpg_file)
"""

def detectTagsFilter(im_opened, refineEdges, decodeSharpening, quadDecimate, quadSigma,
               criticalAngle, maxLineFitMSE, maxNumMaxima, minClusterPixels, minWhiteBlackDiff, CropBoxFactor):
    im = im_opened
    if isinstance(im, np.ndarray) and im.dtype in (np.uint8, np.float32):
        return detectTags(im, refineEdges, decodeSharpening, quadDecimate, quadSigma,
               criticalAngle, maxLineFitMSE, maxNumMaxima, minClusterPixels, minWhiteBlackDiff, CropBoxFactor)
    else:
        print("No image to detect tags on")

def detectTags(img_nb, refineEdges, decodeSharpening, quadDecimate, quadSigma,
               criticalAngle, maxLineFitMSE, maxNumMaxima, minClusterPixels, minWhiteBlackDiff, CropBoxFactor):
    time_first_load_tag = time.time()
    #img_nb = cv2.cvtColor(img_nb, cv2.COLOR_BGR2GRAY)
    print("Detect tags...")
    # New call for AprilTagDetector class
    detector = april.AprilTagDetector()
    
    #Change config
    config = detector.getConfig()
    #print("Default config:")
    #print(config.refineEdges)
    #print(f"decodeSharpening {config.decodeSharpening}, quadDecimate {config.quadDecimate}, quadSigma {config.quadSigma}")
    
    """ refineEdges : When true, the edges of the each quad are adjusted to "snap to" strong gradients nearby. This is useful
    when decimation is employed, as it can increase the quantity of the initial quad estimate substantially.
    Generally (Default) recommended to be on (True).
    Trying False on purpose to not induce False detections of not flat targets. """
    #config.refineEdges = False # True Default
    config.refineEdges = refineEdges
    
    """ decodeSharpening : How much sharpening should be done to decoded images. This can help decode small tags but may or may not
    help in odd lighting conditions or low light conditions. Default is 0.25"""
    #config.decodeSharpening = 0.25 # 0.25 Default
    config.decodeSharpening = decodeSharpening
    
    """ It's the resize factor before detection (not impacting binary decoding (=payload) still full resolution.
    High value = quick but will miss some targets. """
    #config.quadDecimate = 4 # 2 Default
    config.quadDecimate = quadDecimate
    """ NB: with 1 : detects LESS targets !
    With 2, changing slightly the coordinates, a lot more with 4"""
    
    """ quadSigma : What Gaussian blur should be applied to the segmented image (used for quad detection).
    Very noisy images benefit from non-zero values (e.g. 0.8) Default is 0.0 """
    config.quadSigma = quadSigma
    
    # TO SWITCH TO TRUE TO DEGUG ONLY, increases time from 0.7 to 1.9 sec on a 45 MPx image
    #config.debug = True
    detector.setConfig(config)
    
    
    config_quad_threshold = detector.getQuadThresholdParameters()
    
    #print("Default config QuadThreshold:")
    #print(f"criticalAngle {config_quad_threshold.criticalAngle}, qmaxLineFitMSE {config_quad_threshold.maxLineFitMSE}, maxNumMaxima {config_quad_threshold.maxNumMaxima}, minClusterPixels {config_quad_threshold.minClusterPixels}, minWhiteBlackDiff {config_quad_threshold.minWhiteBlackDiff}")
    """ Critical angle : The detector will reject quads where pairs of edges have angles that are close to straight
    or close to 180 degrees. Zero means that no quads are rejected. """
    config_quad_threshold.criticalAngle = criticalAngle # Default pi/4 max pi/2
    
    """ maxLineFitMSE : When fitting lines to the contours, the maximum mean squared error allowed.
    This is useful in rejecting contours that are far from being quad shaped; rejecting these quads "early" saves expensive decoding processing.
    Default is 10.0 """
    config_quad_threshold.maxLineFitMSE = maxLineFitMSE
    
    """ maxNumMaxima : How many corner candidates to consider when segmenting a group of pixels into a quad. Default is 10. """
    config_quad_threshold.maxNumMaxima = maxNumMaxima
    
    """ minClusterPixels : Threshold used to reject quads containing too few pixels. Default is 5 pixels. """
    config_quad_threshold.minClusterPixels = minClusterPixels
    
    """ minWhiteBlackDiff : Minimum brightness offset. When we build our model of black & white pixels, we add an extra check
    that the white model must be (overall) brighter than the black model. How much brighter (in pixel value, [0,255]).
    Default is 5. """
    config_quad_threshold.minWhiteBlackDiff = minWhiteBlackDiff
    
    detector.setQuadThresholdParameters(config_quad_threshold)

    # Type of AprilTag target (here tag36h11)
    detector.addFamily("tag36h11")
    # If not needed to detect this family of tags anymore
    #detector.removeFamily("tag36h11")

    #global tags
    tags = detector.detect(img_nb)
    print(f"{len(tags)} Tags:")
    
    #global refined_tags
    refined_tags = []
    #monge_tags = []
    sobel_tags = []
    #crop_search_zones = [] # Storing in image rather than coordinates, more convenient for layering
    #print(tags)
    
    # Prepare crop zones lines
    cropzones_img = np.zeros((img_nb.shape[0], img_nb.shape[1]), dtype=np.uint8)
    # Prepare Canny lines image
    canny_img = np.zeros((img_nb.shape[0], img_nb.shape[1]), dtype=np.uint8)
    # Prepare Sobel image
    sobel_img = np.zeros((img_nb.shape[0], img_nb.shape[1]), dtype=np.uint8)
    
    for tag in tags:
        tag_id = tag.getId()
        tag_x = tag.getCenter().x
        tag_y = tag.getCenter().y
        print("ID: "+str(tag_id)+", First detection (robotpy_apriltag) Center: "+str(round(tag_x, 2))+", "+str(round(tag_y, 2)))
        #print("ID: "+str(tag_id))
        # Test report file: "a" mode for add
        """with open('Report_april_tag'+str(tag_id)+'.txt', 'a') as f:
            f.write(str(round(tag_x, 1))+", "+str(round(tag_y, 1))+
                    ", refineEdges "+str(config.refineEdges)+
                    ", quadDecimate "+str(config.quadDecimate)+", decodeSharpening "+str(config.decodeSharpening)+
                    ", criticalAngle "+str(config_quad_threshold.criticalAngle)+
                    ", Tags "+str(len(tags))+'\n')
        with open('Report_april_tag'+str(tag_id)+'.txt', 'r') as f:
            for line in f:
                print(line.strip())"""
        
        # In a defined zone, proportionally calculated depending on corners position, around the detected center (TODO)
        # well, fixed zone right now:
        corners_buf = tuple(np.zeros(8, dtype=np.float32))  # Crée un tuple de 8 zéros en float32
        corners = tag.getCorners(corners_buf)
        corner_x_min = np.min(corners[::2])
        corner_x_max = np.max(corners[::2])
        corner_y_min = np.min(corners[1::2])
        corner_y_max = np.max(corners[1::2])
        minvalue = 30 # Rectangle should not be less than that
        search_zone_x = max(minvalue, int(CropBoxFactor * (corner_x_max - corner_x_min)/2)*2) # Don't want a too small zone to calculate gradients
        search_zone_y = max(minvalue, int(CropBoxFactor * (corner_y_max - corner_y_min)/2)*2)
        #print(f"Search zone size for id {tag_id}: {search_zone_x}, {search_zone_y}")
        x_L_c = int(int(round(tag_x,0)) - search_zone_x/2)
        x_R_c = int(int(round(tag_x,0)) + search_zone_x/2) # End bound is an excluded one: [Start:End[ So will add 1 to the end bound
        y_L_c = int(int(round(tag_y,0)) - search_zone_y/2)
        y_R_c = int(int(round(tag_y,0)) + search_zone_y/2) # End bound is an excluded one: [Start:End[ So will add 1 to the end bound
        #print(f"crop_search_zone: {y_L_c}:{y_R_c}, {x_L_c}:{x_R_c}")
        crop_search_zone = img_nb[y_L_c:y_R_c + 1, x_L_c:x_R_c + 1] # End bound is an excluded one: [Start:End[
        #crop_search_zone = cv2.cvtColor(crop_search_zone,cv2.COLOR_BGR2GRAY)
        """ Debug, show crop search zones
        #print(f"crop_search_zone: {y_L_c}:{y_R_c}, {x_L_c}:{x_R_c}")"""
        #crop_search_zones.append([x_L_c, x_R_c, y_L_c, y_R_c]) # Storing in image rather than coordinates, more convenient for layering
        # Draw lines on dedicated layer (on outer pixels) to keep the cropping zone safe for showing
        cv2.line(cropzones_img,(x_L_c - 1, y_L_c - 1),(x_R_c + 1, y_L_c - 1),(80,80,80),1)
        cv2.line(cropzones_img,(x_L_c - 1, y_R_c + 1),(x_R_c + 1, y_R_c + 1),(80,80,80),1)
        cv2.line(cropzones_img,(x_L_c - 1, y_L_c - 1),(x_L_c - 1, y_R_c + 1),(80,80,80),1)
        cv2.line(cropzones_img,(x_R_c + 1, y_L_c - 1),(x_R_c + 1, y_R_c + 1),(80,80,80),1)
        
        """ Step by step then for loop without mask an image generation"""
        canny_center_pixels = [] # preparing mean value by computing array
        canny_mask_done = False
        
        th_steps = 11 # Odd number: 3, 5,... for instance 7 will start from 127-7=[120 and run until 127+7+1=135[
        # th_steps = 9 or 11 is quite good with a step of 2 in range
        # NB: with a step of 2, the loop will run (th_steps+1) times
        for th in range(127 - th_steps, 127 + th_steps + 1, 2): #last excluded
        # [ Gaussian filter with kernel size 9
        # Then normalize and apply threshold to keep median values: [[120-132] > 255]
            norm = on_crop_compute.meanPointPerValueNorm(crop_search_zone, th)
        # Compute Canny contours on normalized hourglass shape
        #print(f"Canny (for id {tag_id})")
            norm = on_crop_compute.canny(norm)
        #print("Size of norm image: "+str(norm.shape[1])+", "+str(norm.shape[0]))
        # Store points from canny image to separated lines, find the closest points from the 2 longest lines and find the center
            canny_lines = on_crop_compute.canny_lines(norm)
            canny_tmp_center = on_crop_compute.canny_contours_to_center(canny_lines) # /!\ Returning coordinates of the pixel, not the center of it.
            #print(canny_tmp_center)
            if canny_tmp_center != False:
                canny_center_pixels.append(on_crop_compute.canny_contours_to_center(canny_lines))
                # Debug/Show bonus: canny image layer with canny lines on it
                if canny_mask_done == False:
                    mask = np.where(norm > 0)
                    canny_img[y_L_c:y_R_c+1, x_L_c:x_R_c+1][mask] = norm[mask]
                    canny_mask_done = True
        #    hough = on_crop_compute.houghlines(norm)
            #hough = cv2.cvtColor(hough,cv2.COLOR_GRAY2BGR)
        #    img[y_L_c:y_R_c, x_L_c:x_R_c, :] = hough

        if np.any(canny_center_pixels):
            #print(canny_center_pixels) # pixels coords of each point
            canny_center_pixel = np.mean(canny_center_pixels, axis=0)
        
            canny_center_coords = (canny_center_pixel + np.array([x_L_c + 0.5, y_L_c + 0.5])) # image coords of mean point
            #print(canny_center_pixel + np.array([x_L_c + 0.5, y_L_c + 0.5]))
            #print(canny_center_coords)
            rtag_x, rtag_y = canny_center_coords
            refined_tags.append([tag_id, rtag_x, rtag_y])
            print(f"Refined center for {tag_id}: "+str(round(rtag_x,2))+", "+str(round(rtag_y,2))+
                  ". Distances "+str(round(rtag_x-tag_x,2))+", "+str(round(rtag_y-tag_y,2)))
            #img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(30,255,30),thickness=thickness)
            
            #refine? NO, too poor ! But Monge ? Canny blackest pixel ?
            """img_origine_nb = cv2.cvtColor(img_origine, cv2.COLOR_BGR2GRAY)
                refined_coordinates = cv2.cornerSubPix(img_origine_nb, np.array([[[tag_x, tag_y]]], dtype=np.float32), subpix_zone, (-1, -1), criteria)
                tag_x, tag_y = refined_coordinates.ravel()
                img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(100,100,255),thickness=thickness)
                print(str(tag_x)+", "+str(tag_y))"""
            
            # [Test Monge
            """margin_crop = 25
            crop_search_zone = img_nb[int(rtag_y)-margin_crop:int(rtag_y)+margin_crop+1,
                                      int(rtag_x)-margin_crop:int(rtag_x)+margin_crop+1]
            #crop_search_zone = cv2.cvtColor(crop_search_zone, cv2.COLOR_BGR2GRAY)"""
            """monge_matrix = on_crop_compute_sobel.monge_positive(crop_search_zone, num_points=3)
            for point in monge_matrix:
                value = point[2]
                monge_x = tag_x - margin_crop + point[0]
                monge_y = tag_y - margin_crop + point[1]
                #print(f"Coordonnées x3 Monge {tag_id}: ({monge_x}, {monge_y}), Valeur : {point[2]}")
                img = drawCross(img,coords=(monge_x,monge_y),length=length,BGR=(228,138,214),thickness=thickness)"""
            """monge_matrix = on_crop_compute_sobel.monge_positive(crop_search_zone, num_points=1)
            for point in monge_matrix:
                #value = point[2]
                monge_x = tag_x - margin_crop + point[0]
                monge_y = tag_y - margin_crop + point[1]
                monge_tags.append([tag_id, monge_x, monge_y])
                #print(f"Coordonnées x1 Monge {tag_id}: ({monge_x}, {monge_y}), Valeur : {point[2]}")
                
                #img = drawCross(img,coords=(monge_x,monge_y),length=length,BGR=(100,55,255),thickness=1)"""
            # Test Monge]
            
            # [Test Canny
            # Test Canny]
            
    print("\nExecution time: "+str(np.round(time.time() - time_first_load_tag,1))+" sec")
    return tags, refined_tags, canny_img, cropzones_img

def find_closest_feature(features, crop_start_x, crop_start_y, X, Y):
    features = np.int32(features).reshape(-1, 2)  # Convert
    dist = np.sqrt((features[:, 0] + crop_start_x - X)**2 + (features[:, 1] + crop_start_y - Y)**2)
    closest_index = np.argmin(dist)
    return closest_index, np.min(dist)

def drawCross(img,coords,length,BGR,thickness):
    size = (img.shape[1], img.shape[0])
    tag_x, tag_y = coords
    x_L = int(tag_x - length/2)
    x_L = 0 if x_L < 0 else size[0]-1 if x_L > size[0] else x_L
    x_R = int(tag_x + length/2)
    x_R = 1 if x_R < 0 else size[0] if x_R > size[0] else x_R
    y_L = int(tag_y - length/2)
    y_L = 0 if y_L < 0 else size[0]-1 if y_L > size[0] else y_L
    y_R = int(tag_y + length/2)
    y_R = 1 if y_R < 0 else size[0] if y_R > size[0] else y_R
    cv2.line(img,(x_L, int(tag_y)),(x_R, int(tag_y)),BGR,thickness)
    cv2.line(img,(int(tag_x), y_L),(int(tag_x), y_R),BGR,thickness)
    return img
    
def drawTags(img_origine, tags, refined_tags, thickness=1, length=np.float32(100)):
    if isinstance(img_origine, np.ndarray) and img_origine.dtype in (np.uint8, np.float32):
        size = (img_origine.shape[1], img_origine.shape[0])
        img = np.zeros((size[1],size[0],1), np.uint8) # 1 Canal
        #img = cv2.bitwise_not(img) # All white
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
        """criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        subpix_zone = (5,5)"""
        
        #draw tags directly out from robotpy_apriltags detector (ie. not refined)
        for tag in tags:
            tag_id = tag.getId()
            tag_x = tag.getCenter().x
            tag_y = tag.getCenter().y
            ###img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(100,100,255),thickness=thickness)
            cv2.putText(img, str(tag_id), (max(int(tag_x - 210), 0),max(int(tag_y - 60), 0)), cv2.FONT_HERSHEY_SIMPLEX, 3, (20,20,255), 5, cv2.LINE_AA)
            # Arguments: img, text, (x,y), font, fontscale, (BGR), thickness, type of line?)
            
            # [Sobel
            """sobel = on_crop_compute_sobel.sob(crop_search_zone)
            sobel = cv2.cvtColor(sobel,cv2.COLOR_GRAY2BGR)
            img[y_L_c:y_R_c, x_L_c+70:x_R_c+70, :] = sobel
            #xf,yf = on_crop_compute_sobel.weighted_center(sobel)
            #xf,yf = on_crop_compute_sobel.contours_to_center(sobel)
            xf,yf = on_crop_compute_sobel.find_average_pixel(sobel,200)
            tag_x = xf + x_L_c
            tag_y = yf + y_L_c
            img = drawCross(img,coords=(tag_x+70,tag_y),length=length,BGR=(255,255,100),thickness=thickness)"""
            # Sobel]
            #_, sobel = cv2.threshold(sobel, 200, 255, cv2.THRESH_BINARY)
            # [Test Monge
            """monge_matrix = on_crop_compute_sobel.monge_positive(crop_search_zone, num_points=3)
            for point in monge_matrix:
                print(f"Coordonnées x3 Monge: ({point[0]}, {point[1]}), Valeur : {point[2]}")
                xf = point[0]
                yf = point[1]
                value = point[2]
                tag_x = xf + x_L_c
                tag_y = yf + y_L_c
                img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(228,138,214),thickness=thickness)
            monge_matrix = on_crop_compute_sobel.monge_positive(crop_search_zone, num_points=1)
            for point in monge_matrix:
                print(f"Coordonnées x1 Monge: ({point[0]}, {point[1]}), Valeur : {point[2]}")
                xf = point[0]
                yf = point[1]
                value = point[2]
                tag_x = xf + x_L_c
                tag_y = yf + y_L_c
                img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(100,55,255),thickness=1)"""
            # Test Monge]
            
            # [Hough
            # Needs another crop region to try here.
                ### Just code copy from previous function detect
            corners_buf = tuple(np.zeros(8, dtype=np.float32))  # Crée un tuple de 8 zéros en float32
            corners = tag.getCorners(corners_buf)
            corner_x_min = np.min(corners[::2])
            corner_x_max = np.max(corners[::2])
            corner_y_min = np.min(corners[1::2])
            corner_y_max = np.max(corners[1::2])
            minvalue = 30 # Rectangle should not be less than that
            CropBoxFactor = .12 # sligth change compared to detector ?
            search_zone_x = max(minvalue, int(CropBoxFactor * (corner_x_max - corner_x_min)/2)*2) # Don't want a too small zone to calculate gradients
            search_zone_y = max(minvalue, int(CropBoxFactor * (corner_y_max - corner_y_min)/2)*2)
            #print(f"Search zone size for id {tag_id}: {search_zone_x}, {search_zone_y}")
            x_L_c = int(int(round(tag_x,0)) - search_zone_x/2)
            x_R_c = int(int(round(tag_x,0)) + search_zone_x/2) # End bound is an excluded one: [Start:End[ So will add 1 to the end bound
            y_L_c = int(int(round(tag_y,0)) - search_zone_y/2)
            y_R_c = int(int(round(tag_y,0)) + search_zone_y/2) # End bound is an excluded one: [Start:End[ So will add 1 to the end bound
            #print(f"crop_search_zone: {y_L_c}:{y_R_c}, {x_L_c}:{x_R_c}")
            crop_search_zone = img_origine[y_L_c:y_R_c + 1, x_L_c:x_R_c + 1] # End bound is an excluded one: [Start:End[
            
            #cv2.cvtColor(crop_search_zone,cv2.COLOR_BGR2GRAY)
            hough = on_crop_compute.houghlines(crop_search_zone)
            mask = np.where(hough > 0)
            #hough = cv2.cvtColor(hough,cv2.COLOR_GRAY2BGR)
            ###img[y_L_c:y_R_c+1, x_L_c:x_R_c+1, :][mask] = hough[mask]
            # Hough]
            # Detection of corners inside the search zone
            # Deconvolve = sharpen
            """ kernel = np.ones((5, 5)) / 25
            crop_search_zone = cv2.filter2D(crop_search_zone, -1, kernel) """
            # features = cv2.goodFeaturesToTrack(crop_search_zone, maxCorners=1, qualityLevel=0.01, minDistance=10) # Basic options
            #features = cv2.goodFeaturesToTrack(crop_search_zone, maxCorners=1, qualityLevel=0.1, minDistance=10)
            #cv.goodFeaturesToTrackWithQuality(image, maxCorners, qualityLevel, minDistance, mask
            #[, corners[, cornersQuality[, blockSize[, gradientSize[, useHarrisDetector[, k]]]]]]
            """Parameters
    image : Input 8-bit or floating-point 32-bit, single-channel image.
    corners : Output vector of detected corners.
    maxCorners : Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
    qualityLevel : Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris function response (see cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
    minDistance : Minimum possible Euclidean distance between the returned corners.
    mask : Optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
    blockSize : Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See cornerEigenValsAndVecs .
    useHarrisDetector : Parameter indicating whether to use a Harris detector (see cornerHarris) or cornerMinEigenVal.
    k : Free parameter of the Harris detector."""
            """features = cv2.goodFeaturesToTrack(crop_search_zone, maxCorners=3, qualityLevel=0.01, minDistance=10, useHarrisDetector = False)
            print(features)
            if features is not None:
                for f in features:
                    print(f)
                    xf,yf = f.ravel()
                    tag_x = xf + x_L_c
                    tag_y = yf + y_L_c
                    #img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(255,100,100),thickness=thickness)
                    print("Feature: "+str(tag_x)+", "+str(tag_y))
                closest_id, dist = find_closest_feature(features, crop_start_x=x_L_c, crop_start_y=y_L_c, X=tag.getCenter().x, Y=tag.getCenter().y)
                xf,yf = features[closest_id].ravel()
                tag_x = xf + x_L_c
                tag_y = yf + y_L_c
                img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(255,255,100),thickness=thickness)
                print("Closest: "+str(tag_x)+", "+str(tag_y)+" dist "+str(round(dist,2))+" px")"""
                #refine?
            """img_origine_nb = cv2.cvtColor(img_origine, cv2.COLOR_BGR2GRAY)
                    refined_corners = cv2.cornerSubPix(img_origine_nb, np.array([[[tag_x, tag_y]]], dtype=np.float32), subpix_zone, (-1, -1), criteria)
                    tag_x, tag_y = refined_corners.ravel()
                    img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(100,255,100),thickness=thickness)
                    print(str(tag_x)+", "+str(tag_y))"""
            
            
            """coords = on_crop_compute.meanPointPerValue(crop_search_zone)
            for xf, yf in zip(coords[0], coords[1]):
                tag_x = xf + x_L_c
                tag_y = yf + y_L_c
                img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(255,100,100),thickness=thickness)"""
            
            
            
            """features_h = cv2.cornerHarris(crop_search_zone, blockSize=10, ksize=11, k=0.1)
            threshold = 0.8 * features_h.max()
            print(threshold)
            if features_h is not None:
                corner_locations = np.where(features_h > threshold)
                corner_coordinates = list(zip(corner_locations[0], corner_locations[1]))
                for f in corner_coordinates:
                    xf,yf = f
                    tag_x = xf + x_L_c
                    tag_y = yf + y_L_c
                    img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(255,100,100),thickness=thickness)
                    print(f"Corner Harris : {xf}, {yf}")"""
            """refined_corners = cv2.cornerSubPix(img_origine_nb, np.array([[[tag_x, tag_y]]], dtype=np.float32), subpix_zone, (-1, -1), criteria)
            tag_x = refined_corners[0][0][0]
            tag_y = refined_corners[0][0][1]"""
                    
        for tag in refined_tags:
            tag_id = tag[0]
            tag_x = tag[1]
            tag_y = tag[2]
            img = drawCross(img,coords=(tag_x,tag_y),length=length,BGR=(30,255,30),thickness=thickness)
            #cv2.putText(img, str(tag_id), (max(int(tag_x - 210), 0),max(int(tag_y - 60), 0)), cv2.FONT_HERSHEY_SIMPLEX, 3, (30,255,30), 5, cv2.LINE_AA)

            # [Test Monge
            """margin_crop = 15
            crop_search_zone = img_origine[int(tag_y) - margin_crop : int(tag_y) + margin_crop + 1,
                                           int(tag_x) - margin_crop : int(tag_x) + margin_crop + 1]
            crop_search_zone = cv2.cvtColor(crop_search_zone, cv2.COLOR_BGR2GRAY)
            monge_matrix = on_crop_compute_sobel.monge_positive(crop_search_zone, num_points=3)
            for point in monge_matrix:
                value = point[2]
                monge_x = tag_x - margin_crop + point[0]
                monge_y = tag_y - margin_crop + point[1]
                #print(f"Coordonnées x3 Monge {tag_id}: ({monge_x}, {monge_y}), Valeur : {point[2]}")
                img = drawCross(img,coords=(monge_x,monge_y),length=length,BGR=(228,138,214),thickness=thickness)
            monge_matrix = on_crop_compute_sobel.monge_positive(crop_search_zone, num_points=1)
            for point in monge_matrix:
                value = point[2]
                monge_x = tag_x - margin_crop + point[0]
                monge_y = tag_y - margin_crop + point[1]
                print(f"Coordonnées x1 Monge {tag_id}: ({monge_x}, {monge_y}), Valeur : {point[2]}")
                img = drawCross(img,coords=(monge_x,monge_y),length=length,BGR=(100,55,255),thickness=1)"""
            # Test Monge]
            
        return img

#detectTags(img)