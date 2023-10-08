"""
    Created on Monday May 29 2023 18:24 GMT-6
    Autor: Alberto Estudillo Moreno
    Last Modification: Monday June 5
    
    Using autobbox V1.7, rastreo V1.8 and InitialFrame V0.13
"""

import cv2, time
import pandas as pd
import numpy as np
import exiftool as etl
import matplotlib.pyplot as plt

# ---------------- Funciones por modulo ---------------- #

#############################
#### Initial Frame v0.13 ####
#############################

class InitialFrame:  
    
    def lightIntensity(path, percent=float):
        capture = cv2.VideoCapture(path)
        while(capture.isOpened()):
            ret, frame = capture.read()
            fps = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            
            if ret == True:
  
             #Convertimos a escala de grises y obtenemos un histograma...
             #...para calcular que tanta intensidad de luz hay en el frame
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #calcHist([imagen],[canal(es)], mask, [bins], [rangomin,rangomax+1])
             histogram = cv2.calcHist([gray],[0], None, [256], [0,256])
             #Sumamos el numero de pixeles oscuros en el rango de [0,35]
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
        #Imagen auxiliar para determinara el area de los puntos...
        #...donde se usara el detector de movimiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_image = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        aux_image = cv2.drawContours(aux_image, [area_points], -1, (255), -1)
        image_area = cv2.bitwise_and(gray, gray, mask= aux_image)
        
        return image_area
    
    def morphologicTransform(image, kernel, iter=int):
        #Se aplican transformaciones morfologicas para mejorar la imagen binaria
        img_mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        img_dil = cv2.dilate(img_mask, None, iterations=iter)
        demos = cv2.demosaicing(img_dil, code= cv2.COLOR_BayerBG2GRAY)
        bn = cv2.inRange(demos, np.array([25]), np.array([255]))
        
        return bn
    
    def circleDetection(image, blur):
        #Busca la particula con deteccion de circulos
        blur_image = cv2.medianBlur(image, blur)
        particle = cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT_ALT, 1, 2000, param1= 50, param2= 0.85, minRadius=6, maxRadius=30)
        if particle is not None:
            (x,y,r) = np.around((np.around(particle[0,:], decimals=6).astype('float'))[0,:], decimals=6).astype('float')
            
        if particle is None:
            x,y,r = 0,0,0
        
        return (x,y,r)
    
################################
#### Auto Bounding Box v1.7 ####
################################

class BoundinBox:
 def get_auto_bbox(path, initial_frame, minimal_contrast, delta=int, show=False):
    
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)

    success, frame = capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, np.array([minimal_contrast]), np.array([255]))
    blur = cv2.medianBlur(mask, 5)
    
    estimate = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT_ALT, 1, 2000, param1=50, param2=0.85, minRadius=5, maxRadius=30)
    estims = np.round(estimate[0,:]).astype('int')
    estim = np.round(estims[0,:]).astype('int')
    (a,b,r) = estim
    print(estim)
    blur[:,:]=0
    circle_bn = cv2.circle(blur, (a,b), r, (255, 255, 255), 2)

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

#####################
#### Rastreo 1.8 ####
#####################

class rastreoCSRT:
 def get_rotation(video_path):
    ########## Se comprueba la orientacion del video #########
    with etl.ExifToolHelper() as et:
        metadata = pd.DataFrame(et.get_tags(video_path,tags="Rotation"))
    orientation = int(metadata['Composite:Rotation'][0])
    
    return orientation

 def traking_particle_CSRT(video_path, frame_ini, frame_end, orientation, bbox, irl=False):
    # Arreglos
    coords = []
    
    ########## Colocar video en frame inicial #########
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
    success, frame = capture.read()
    
    
    ######### Escoger Metodo de Tracking ##########
    # En este caso se usa el metodo CSRT, ya que es mas preciso y rapido...
    #... que otros metodos existentes en la libreria de OpenCV #
    tracker = cv2.TrackerCSRT_create()

    ######### Iniciar tracking con bbox ##########
    success = tracker.init(frame,bbox)

    while True:
        ######### Calculando punto central de bbox#########
        X = int((bbox[0]+bbox[0]+bbox[2])/2)
        Y = int((bbox[1]+bbox[1]+bbox[3])/2)
        
        ######## Se guardan las coordenadas dependiendo de la orientacion ############
        if orientation == 90:
            coords.append([Y,X])
        
        else:
            coords.append([X,Y])
        
        ######### Frame actual #########
        fps = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        success_frame, frame = capture.read()

        if fps == frame_end:
            break
    
        if not success_frame:
            print('No hay mas frames')
            capture.release()
            cv2.destroyAllWindows()
            break
        
        ####### Actualizando el tracker CSRT #########
        success, bbox = tracker.update(frame)
        
        if irl == True:
         if success:
            ########## Puntos de rastreo ########
            pts=np.array(coords, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed= False, color=(0,0,255), thickness=2) 
            
            ########## Bounding Box ##########
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,255,255), 2, 1)
        
            ########## Mostrar imagen #########
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tracking', 1280,720)
            cv2.putText(frame, 'Tracker', (400,20), font, 1, (0,255,0), 2)
            cv2.putText(frame, str(fps), (300,20), font, 1, (0,255,0), 2)
            cv2.imshow('Tracking',frame)    
            cv2.waitKey(1)
        
        if success is not True:
            print('Se detecto un error al rastrear')
            break
    return coords


