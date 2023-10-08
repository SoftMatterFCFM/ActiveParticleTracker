"""
Created on Saturday March 25 2023 17:16 GMT-6
Autor:Alberto Estudillo Moreno 
Last modification: Wednesday Sep 22:57 GMT-6

"""

import numpy as np
import cv2
import exiftool as etl
import time, pandas as pd

def get_auto_bbox(frame, delta: int, orientation: int, area_points, show: bool =False):
    """ Find a region where the particle is located in the initial frame.

    Args:
        frame : Initial frame of the video to be tracking.
        delta (int): Increase size of region where the particle is located.
        orientation (int): Integer that describes whether the video is horizontal (0) 
        or vertical (90).
        area_points: Array of points that limit the search area to avoid noise.
        show (bool, optional): Shows an image of the found region. Defaults to 'False'.

    Returns:
        (x, y, w, h): Returns a point (x,y), the width and height of the found region.
        Where (x, y) are the coordinate from top left corner.
    """    
    # Check the video orientation to fix...
    #... the region of interes given by area_points.
    if orientation == 90:
        inverse = []

        for point in area_points:
            new_point = point[::-1]
            inverse.append(new_point)
    
        area_points = np.array(inverse)
    
    # Creates a specific region to search the particle and ...
    #... avoid as much noise as possible.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_image = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    aux_image = cv2.drawContours(aux_image, [area_points], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask= aux_image)
    # Converts gray scale to binary scale to reduce noise...
    #... around the particle.
    bn = cv2.inRange(image_area, np.array([15]), np.array([255]))
    # Converts the binary scale to gray scale with...
    #... this condition "if pixel => 1 -> pixel = 255...
    #... and pixel = 0 -> pixel = 0".
    bn[bn>=1] = 255

    # Finds circle that fits the particle. Center (x,y) and radius.
    estimate = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT_ALT, 1, 2000, param1=50, 
                                param2=0.85, minRadius=5, maxRadius=30)
    estims = np.round(estimate[0,:]).astype('int')
    estim = np.round(estims[0,:]).astype('int')
    (a,b,r) = estim
    # Create a full black image to draw the found circle...
    #... and thus be able to estimate the bbox.
    bn[:,:]=0
    circle_bn = cv2.circle(bn, (a,b), r, (255, 255, 255), 2)

    bbox, check = cv2.findContours(circle_bn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(bbox[0])
    
    x1=x-delta
    y1=y-delta
    w=w+delta
    h=h+delta
    
    if show == True:
        rect = cv2.rectangle(circle_bn, (x1,y1),(x+w,y+h), (255,255,255), 2, 1)
        cv2.imshow('bbox', rect); cv2.waitKey(0)
    
    return x1,y1,w,h

########## Getting time ##########
initial_time = time.time()

########## Access paths #########
pathvid = 'F:\\VParticles\\Puerta28\\'

########## Parameters ######### 
namevid = '6-GX010249-8punto18.MP4' 
frame_ini = 206
area_points = np.array([[1201,697],[1201,381],[767,381],[767,697]]) 
path = pathvid + namevid

#------------- Main Code --------------#
capture = cv2.VideoCapture(path)
capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
success, frame = capture.read()
with etl.ExifToolHelper() as et:
    metadata = pd.DataFrame(et.get_tags(path,tags="Rotation"))
orientation = int(metadata['Composite:Rotation'][0])

bbox= get_auto_bbox(frame, 3, orientation, area_points, True)
capture.release()
print(bbox)
