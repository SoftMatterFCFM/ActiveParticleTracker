"""
Created on Sunday April 16 2023 23:16 GMT-6
Autor:Alberto Estudillo Moreno 
Last modification: Thursday September 14 12:33 GMT-6

"""

import cv2
import numpy as np
import pandas as pd
import exiftool as etl


def lightIntensity(path, percent: float = 0.0):
    """ Locate first frame with lowest luminosity according to the darkness percent.

    Args:
        path (str): Path of the video.
        percent (float, optional): Required drakness percent to satisfy the
        luminosity conditions. Defaults to "0.0".

    Returns:
        fps (int): Returns the number of the frame that satisfy the
        luminosity conditions.
    """    
    capture = cv2.VideoCapture(path)
    while(capture.isOpened()):
        ret, frame = capture.read()
        fps = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            
        if ret == True:
  
         # Converts to gray scale and obtain the histogram to...
         #...compute how much light intensity has the current frame.
         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         #calcHist([image],[channel(s)], mask, [bins], [minrange,maxrange+1])
         histogram = cv2.calcHist([gray],[0], None, [256], [0,256])
         # Compute total number of dark pixels in the range [0,35]
         darkness = sum(histogram[0:35])
        
         if darkness > (1920*1080)*percent:
            print('A partir del frame %i' %fps + ' el video tiene poca iluminacion.')
            capture.release()
            cv2.destroyAllWindows()
            break
            
         else:
            continue
        
    return fps
    
def auxiliarImage(frame, area_points):
    """ Auxiliar image to determine the area of the points where
    the motion detector will be used.

    Args:
        frame: Frame of the video.
        area_points: Array of points that limits the image area to avoid noise.

    Returns:
        image_area: Auxiliar image with the selected region of interest.
    """    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_image = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    aux_image = cv2.drawContours(aux_image, [area_points], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask= aux_image)
        
    return image_area
    
def morphologicTransform(image, kernel):
    # Apply morphological transformations to enhance the binary image.
    img_mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img_dil = cv2.dilate(img_mask, None, iterations= 1)
    demos = cv2.demosaicing(img_dil, code= cv2.COLOR_BayerBG2GRAY)
    bn = cv2.inRange(demos, np.array([25]), np.array([255]))
        
    return bn
    
def movementDetector(path: str, fps: int, area_points, kernel):
    """ Motion detector to find in which frame the particle starts its trail.
    Frame superposition is used to compute the motion according the particle
    area.

    Args:
        path (str): Path of the video.
        fps (int): Frame number with low luminosity.
        area_points: Array of points that limits the image area to avoid noise.
        kernel: Kernel to generate elliptical/circular structures.

    Returns:
        init_frame (int): Returns the frame number that will be the initial
        frame to start tracking the particle.
    """    
    # Auxiliar arrays
    frames = []
    frames_count = []

    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, fps)
        
    while(capture.isOpened()):
        ret , frame = capture.read()
        fps = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            
        if ret == True:
            aux_img = auxiliarImage(frame, area_points)
            img_mask = morphologicTransform(aux_img, kernel)
            # cv2.imshow('mask', img_mask); cv2.waitKey(0)
 
            # Append the frame in the auxiliar array.
            frames.append(img_mask)
            # Append the frame number that is being added in the previous array.
            frames_count.append(fps)
        
            # Make the frame superposition when the auxiliar array has size...
            #... of 5 elements.
            if len(frames) == 5:
                sum_frames = sum(frames)
                # Find the contour of the frame where the 5 frames where superpositioned.
                contour, _ = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Compute the frame area resulting from the superposition.
                for cont in contour:
                    contour_area = cv2.contourArea(cont)
                    print(contour_area)
                
                # If the computed area don't surpass 2.5 times the size of the particle...
                #... it delete the first added frame and the frame number.
                if contour_area < min_area:
                    del frames[0]
                    del frames_count[0]
                    continue
            
                else:
                    cv2.imshow('suma', sum_frames); cv2.waitKey(0)
                    print('Se detecto movimiento entre estos frames: ')
                    print(frames_count)
                    break
            
        else: 
            continue
    
    capture.release()   
    init_frame = frames_count[2]
    return init_frame


#------------ Main Code ------------#
# Access paths.
path_vid = 'F:\\VParticles\\Puerta28\\'
vid_name = '6-GX010249-8punto18.MP4'
full_path = path_vid + vid_name
show_histogram = False
##################################
# Parameters.
percent = 0.97
# Minimum area of superposition to detect motion.
min_area = 1200
# Initial area of superposition.
contour_area = 0

##################################
# Video orientation.
##################################
with etl.ExifToolHelper() as et:
    metadata = pd.DataFrame(et.get_tags(path_vid + vid_name, tags="Rotation"))
orientation = int(metadata['Composite:Rotation'][0])

##################################
# Get the frame with low luminosity.
##################################
fps = lightIntensity(full_path, percent)

###############################################
# Motion detection in specific area.
###############################################
# Parameters.
# Kernel to generate elliptical/circular structures.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# Region of interest, it will be a rectangular area centered...
#... in the frame.
area_points = np.array([[1201,697],[1201,381],[767,381],[767,697]])

# Fix the coordinates according to the video orientation.
if orientation == 90:
    inverse = []

    for point in area_points:
        new_point = point[::-1]
        inverse.append(new_point)
    
    area_points = np.array(inverse)
    
frame_inicial = movementDetector(path_vid + vid_name, fps, area_points, kernel)
print('El frame inicial del video es: '+ str(frame_inicial))