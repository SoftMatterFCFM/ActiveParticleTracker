"""
Created on Monday March 20 2023 23:16 GMT-6
Autor:Alberto Estudillo Moreno 
Last modification: Thursday March 28 12:25 GMT-6

Using autobbox V1.8
"""

import numpy as np
import pandas as pd
import cv2
import exiftool as etl
import time
import matplotlib.pyplot as plt


def get_rotation(video_path):
    ########## Check the video orientation #########
    with etl.ExifToolHelper() as et:
        metadata = pd.DataFrame(et.get_tags(video_path,tags="Rotation"))
    orientation = int(metadata['Composite:Rotation'][0])
    
    return orientation

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

def traking_particle_CSRT(orientation, area, delta: int = 0, irl = False):
    """ Track the particle using CSRT method, which is based in a located
    object inside a region of interest (ROI). This tracking method has a 24 
    frames per second, according to computer limitations.

    Args:
        bbox (tuple): Area data where the particle is located.\n
        orientation (int): Integer that describes whether the video 
        is horizontal (0) or vertical (90).\n
        delta (int): Increase size of region where the particle is 
        located..\n
        irl (bool, optional): Shows a real-time tracking. Defaults to False.

    Returns:
        list: Returns the particles's coordinates array.
        (All data are integer numbers).
    """    
    
    ########## Set initial frame video #########
    capture = cv2.VideoCapture(pathvid+namevid)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
    success, frame = capture.read()
    
    ######### Region of interest is gotten #########
    bbox = get_auto_bbox(frame, delta, orientation, area)
    
    ######### Create tracking method ##########
    # We used the CSRT method, which is more reliable...
    #... than other methods in OpenCV library.
    tracker = cv2.TrackerCSRT_create()

    ######### Start tracking with the ROI ##########
    success_track = tracker.init(frame,bbox)

    while(capture.isOpened()):
        ######### Compute the center of ROI#########
        X = int((bbox[0]+bbox[0]+bbox[2])/2)
        Y = int((bbox[1]+bbox[1]+bbox[3])/2)
        
        ######## Append the coordinates according to video orientation ############
        if orientation == 90:
            coords.append([Y,X])
        
        else:
            coords.append([X,Y])
        
        ######### Current frame #########
        fps = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        success_frame, frame = capture.read()

        if success_frame == True:
          if fps == frame_end:
            break

          ####### Update CSRT tracking #########
          success_track, bbox = tracker.update(frame)
          
          ####### Shows the tracking in real-time ######
          if success_track is True:
            if irl == True:
              ########## Tracking points ########
              pts=np.array(coords, dtype=np.int32)
              cv2.polylines(frame, [pts], isClosed= False, color=(0,0,255), thickness=2) 
            
              ########## ROI ##########
              p1 = (int(bbox[0]), int(bbox[1]))
              p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
              cv2.rectangle(frame, p1, p2, (255,255,255), 2, 1)
        
              ########## Shows image #########
              font = cv2.FONT_HERSHEY_SIMPLEX
              cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
              cv2.resizeWindow('Tracking', 1280,720)
              cv2.putText(frame, 'Tracker', (400,30), font, 1, (0,255,0), 2)
              cv2.putText(frame, str(fps), (300,30), font, 1, (0,255,0), 2)
              cv2.imshow('Tracking',frame)    
              cv2.waitKey(1)
            
            continue
        
          if success_track is not True:
            print('Se detecto un error al rastrear')
            break
        
        else:
            break
    
    print('No hay mas frames')
    capture.release()
    cv2.destroyAllWindows()
    
    return coords

def show_tracking(points):
    """Shows a graph with the completed tracking.

    Args:
        points (array): Array of points.
    """    

    plt.figure('rastreo')
    plt.plot(points[:,0],points[:,1])
    plt.show()
    return 0

########## Getting time ##########
initial_time = time.time()

########## Paths ######### 
pathvid = 'F:\\VParticles\\Puerta28\\'
pathdata = 'G:\\TrackingCompleto\\Resultados\\P28\\'

########## Parameters #########
# Video name #
namevid = '85-GH010339.MP4'
# Initial frame #
frame_ini = 415
# Ending frame for tracking, leave at 0...
#... if tracking entire video is required.
frame_end = 0
delta = 4
area_points = np.array([[1201,697],[1201,381],[767,381],[767,697]]) 
# Array where coordinates will be added #
coords = []

######## Calls the tracking function #########
rotation = get_rotation(pathvid+namevid)
rastreo = traking_particle_CSRT(rotation, area_points, delta, True)

######## Save tracking data ##########
coordinates = np.array(rastreo)
np.savetxt(pathdata+'Tracking'+namevid[0:2]+'.dat', coordinates)

######## Getting time #########
final_time = time.time()
print('El tracking ' +namevid[0:2]+ ' tardo: %.2f'%((final_time-initial_time)/60)+' minutos.')

######## Shows the tracking. #########
show_tracking(coordinates)


