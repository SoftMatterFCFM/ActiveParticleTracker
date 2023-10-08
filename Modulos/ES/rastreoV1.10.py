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
    ########## Se comprueba la orientacion del video #########
    with etl.ExifToolHelper() as et:
        metadata = pd.DataFrame(et.get_tags(video_path,tags="Rotation"))
    orientation = int(metadata['Composite:Rotation'][0])
    
    return orientation

def get_auto_bbox(frame, delta: int, orientation: int, area_points, show: bool =False):
    """ Encuentra una region cerrada donde se encuentra la particula
    en el inicio del video.

    Args:
        frame : Fotograma inicial del video a analizar para rastreo.
        delta (int): Aumenta el tamaño de la region cerrada donde se encuentra la particula.
        orientation (int): Valor que describe si el video esta horizontal(0) o vertical(90).
        area_points (_type_): Puntos dentro de un arreglo que limitan el area de busqueda
        para evitar ruido.
        show (bool, optional): Muestra una imagen de la region encontrada. Predeterminado en 'False'.

    Returns:
        (x, y, w, h): Regresa una coordenada (x,y), el ancho y alto de la region encontrada.
    """    
    #Comprueba la orientacion del video para corregir...
    #... el area de interes dada por area_points.
    if orientation == 90:
        inverse = []

        for point in area_points:
            new_point = point[::-1]
            inverse.append(new_point)
    
        area_points = np.array(inverse)
    
    # Se crea un area especifica para buscar la particula y evitar el mayor ruido posible
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_image = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    aux_image = cv2.drawContours(aux_image, [area_points], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask= aux_image)
    # Se pasa de la escala de grises a escala binaria para reducir el ruido alrededor...
    #... de la particula.
    bn = cv2.inRange(image_area, np.array([15]), np.array([255]))
    # Convierte la escala binaria a escala de grises con..
    #... la conversion de -> pixel = 1 entonces pixel = 255.
    bn[bn>=1] = 255

    estimate = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT_ALT, 1, 2000, param1=50, 
                                param2=0.85, minRadius=5, maxRadius=30)
    estims = np.round(estimate[0,:]).astype('int')
    estim = np.round(estims[0,:]).astype('int')
    (a,b,r) = estim
    # Se crea una imagen completamente en negro para...
    #... dibujar el circulo encontrado y estimar mejor la bbox.
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
    """Rastrea una particula utilizando el metodo CSRT, que se basa
    en un objeto localizado dentro de una region de interes (bbox). Este metodo
    de rastreo tiene una velocidad de 24 frames por segundo, dependiendo de 
    las limitaciones de la computadora.

    Args:
        bbox (tuple): Datos del area donde esta la particula.\n
        orientation (int): Valor para determinar si el video
        esta en formato vertical (90) u horizontal (0).\n
        delta (int):Numero entero para aumentar tamaño de la bbox
        en los 4 ejes. 0 es el numero por defecto.\n
        irl (bool, optional): Muestra el rastreo en tiempo real. Defaults to False.

    Returns:
        list: Regresa un arreglo con las coordenadas de la particula
        (Todos los datos son numeros enteros).
    """    
    
    ########## Colocar video en frame inicial #########
    capture = cv2.VideoCapture(pathvid+namevid)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
    success, frame = capture.read()
    
    ######### Obtiene la region de interes #########
    bbox = get_auto_bbox(frame, delta, orientation, area)
    
    ######### Escoger Metodo de Tracking ##########
    # En este caso se usa el metodo CSRT, ya que es mas preciso y rapido...
    #... que otros metodos existentes en la libreria de OpenCV #
    tracker = cv2.TrackerCSRT_create()

    ######### Iniciar tracking con bbox ##########
    success_track = tracker.init(frame,bbox)

    while(capture.isOpened()):
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

        if success_frame == True:
          if fps == frame_end:
            break

          ####### Actualizando el tracker CSRT #########
          success_track, bbox = tracker.update(frame)
          
          ####### Muestra el rastreo en tiempo real ######
          if success_track is True:
            if irl == True:
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
    """Muestra un grafico del rastreo ya finalizado.

    Args:
        points (array): Arreglo de coordenadas del rastreo.
    """    

    plt.figure('rastreo')
    plt.plot(points[:,0],points[:,1])
    plt.show()
    return 0

########## Toma de tiempo ##########
initial_time = time.time()

########## Rutas de acceso y guardado ######### 
pathvid = 'F:\\VParticles\\Puerta28\\'
pathdata = 'G:\\TrackingCompleto\\Resultados\\P28\\'

########## Parametros #########
# Nombre del video #
namevid = '85-GH010339.MP4'
# Frame inicial #
frame_ini = 415
# Frame de finalizacion de rastreo, dejar en 0...
#... si se quiere rastrear todo el video #
frame_end = 0
# Pixeles que daran un tamaño extra a la bbox...
#... que encerrara a la particula, se aumenta en los 4 ejes #
delta = 4
# Area donde se ubicara la particula en el frame inicial
area_points = np.array([[1201,697],[1201,381],[767,381],[767,697]]) 
# Arreglo donde se guardaran las coordenadas #
coords = []

######## Se llama la funcion para tracking #########
rotation = get_rotation(pathvid+namevid)
rastreo = traking_particle_CSRT(rotation, area_points, delta, True)

######## Guardado de datos del tracking ##########
coordinates = np.array(rastreo)
np.savetxt(pathdata+'Tracking'+namevid[0:2]+'.dat', coordinates)

######## Toma de tiempo #########
final_time = time.time()
print('El tracking ' +namevid[0:2]+ ' tardo: %.2f'%((final_time-initial_time)/60)+' minutos.')

######## Muestra el rastreo completo #########
show_tracking(coordinates)


