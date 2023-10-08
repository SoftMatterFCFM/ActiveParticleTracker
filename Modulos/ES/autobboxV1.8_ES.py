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
    """ Encuentra una region donde se ubica la particula en el fotograma inicial.

    Args:
        frame : Fotograma inicial del video para rastreo.
        delta (int): Aumenta el tamaño de la region donde se encuentra la particula.
        orientation (int): Valor que describe si el video esta horizontal(0) o vertical(90).
        area_points (_type_): Puntos dentro de un arreglo que limitan el area de busqueda
        para evitar ruido.
        show (bool, optional): Muestra una imagen de la region encontrada. Predeterminado en 'False'.

    Returns:
        (x, y, w, h): Regresa una coordenada (x,y), el ancho y alto de la region encontrada.
        Donde (x, y) son las coordenadas de la esquina superior izquierda.
    """    
    #Comprueba la orientacion del video para corregir...
    #... el area de interes dada por area_points.
    if orientation == 90:
        inverse = []

        for point in area_points:
            new_point = point[::-1]
            inverse.append(new_point)
    
        area_points = np.array(inverse)
    
    # Se crea un area especifica para buscar la particula y evitar el mayor ruido posible.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_image = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    aux_image = cv2.drawContours(aux_image, [area_points], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask= aux_image)
    # Se pasa de la escala de grises a escala binaria para reducir el ruido alrededor...
    #... de la particula.
    bn = cv2.inRange(image_area, np.array([15]), np.array([255]))
    # Convierte la escala binaria a escala de grises con..
    #... la conversion de -> pixel = 1 entonces pixel = 255...
    #... y pixel = 0 -> pixel = 0.
    bn[bn>=1] = 255

    # Encuentra el circulo que describe la particula. Centro (x,y) y radio.
    estimate = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT_ALT, 1, 2000, param1=50, 
                                param2=0.85, minRadius=5, maxRadius=30)
    estims = np.round(estimate[0,:]).astype('int')
    estim = np.round(estims[0,:]).astype('int')
    (a,b,r) = estim
    # Se crea una imagen completamente en negro para dibujar el...
    #...circulo encontrado y asi poder estimar mejor la bbox.
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

########## Toma de tiempo ##########
initial_time = time.time()

########## Rutas de acceso #########
pathvid = 'F:\\VParticles\\Puerta28\\'

########## Parametros #########
namevid = '6-GX010249-8punto18.MP4' 
frame_ini = 206
area_points = np.array([[1201,697],[1201,381],[767,381],[767,697]]) 
path = pathvid + namevid

#------------- Codigo Principal --------------#
capture = cv2.VideoCapture(path)
capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
success, frame = capture.read()
with etl.ExifToolHelper() as et:
    metadata = pd.DataFrame(et.get_tags(path,tags="Rotation"))
orientation = int(metadata['Composite:Rotation'][0])

bbox= get_auto_bbox(frame, 3, orientation, area_points, True)
capture.release()
print(bbox)
