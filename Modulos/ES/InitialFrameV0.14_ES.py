"""
Created on Sunday April 16 2023 23:16 GMT-6
Autor:Alberto Estudillo Moreno 
Last modification: Thursday September 14 12:33 GMT-6

"""

import cv2
import numpy as np
import pandas as pd
import exiftool as etl


def lightIntensity(path, percent:float = 0.0):
    """Ubica el primer fotograma con menor luminosidad segun el porcentaje de oscuridad.

    Args:
        path (str): Ruta de acceso del video.
        percent (float, optional): Porcentaje de oscuridad requerido para
        satisfacer las condiciones de luminosidad. Predeterminado con "0.0".

    Returns:
        fps (int): Regresa el numero del fotograma que cumple las condiciones de
        luminosidad.
    """    
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
    """ Imagen auxiliar para determinar el area de los puntos donde
    se utilizara el detector de movimiento.

    Args:
        frame: Fotograma del video.
        area_points: Arreglo de puntos que limitan el area de la imagen
        para reducir el ruido.

    Returns:
        image_area: Imagen auxiliar con la region de interes seleccionada.
    """    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_image = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    aux_image = cv2.drawContours(aux_image, [area_points], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask= aux_image)
        
    return image_area
    
def morphologicTransform(image, kernel):
    #Se aplican transformaciones morfologicas para mejorar la imagen binaria
    img_mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img_dil = cv2.dilate(img_mask, None, iterations= 1)
    demos = cv2.demosaicing(img_dil, code= cv2.COLOR_BayerBG2GRAY)
    bn = cv2.inRange(demos, np.array([25]), np.array([255]))
        
    return bn
    
def movementDetector(path: str, fps: int, area_points, kernel):
    """ Detector de movimiento para encontrar en que fotograma la particula
    empieza su recorrido. Se usa la superposicion de fotogramas para
    calcular el movimiento dependiendo del area de la particula.

    Args:
        path (str): Ruta de acceso al video.
        fps (int): Numero de fotograma con poca luminosidad
        area_points: Arreglo de puntos que limitan el area de la imagen
        para reducir el ruido.
        kernel: Kernel para generar estructuras de forma eliptica/circular

    Returns:
        init_frame (int): Regresa el numero del fotograma que sera nuestro
        fotograma incial para iniciar el rastreo de la particula.
    """    
    #Listas de apoyo
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
 
            #Guarda el frame en una lista
            frames.append(img_mask)
            #Guarda el numero de frame que se esta guardando en la lista anterior
            frames_count.append(fps)
        
            #Superpone los frames guardados cuando la lista tiene un tamaño de 5
            if len(frames) == 5:
                sum_frames = sum(frames)
                #Encuentra el contorno del frame donde se sumaron los 5 frames
                contour, _ = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                #Calcula el area del frame resultante de la suma
                for cont in contour:
                    contour_area = cv2.contourArea(cont)
                    print(contour_area)
                
                #Si el area dentro del contorno no supera los 1400 pixeles, se detiene el calculo...
                #...y se avanza con el siguiente ciclo del While
                if contour_area < min_area:
                #Borra el primer frame guardado en las listas
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


##################################
# Parametros para leer la ubicacion del archivo
path_vid = 'F:\\VParticles\\Puerta28\\'
vid_name = '6-GX010249-8punto18.MP4'
full_path = path_vid + vid_name
show_histogram = False
##################################
#Parametro adicionales
percent = 0.97
#Area minima de la superposicion para detectar movimiento
min_area = 1200
# Area inicial de la superposicion
contour_area = 0

##################################
# ORIENTACION DEL VIDEO.
##################################
with etl.ExifToolHelper() as et:
    metadata = pd.DataFrame(et.get_tags(path_vid + vid_name, tags="Rotation"))
orientation = int(metadata['Composite:Rotation'][0])

##################################
#DEFINE EL FRAME CON BAJA ILUMINACION
##################################
fps = lightIntensity(full_path, percent)

###############################################
# DETECCION DE MOVIMIENTO EN UN AREA ESPECIFICA.
###############################################
#Parametros
#Kernel para generar estructuras de forma eliptica/circular
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#Creamos los puntos del area, sera un rectangulo centrado en la imagen.
area_points = np.array([[1201,697],[1201,381],[767,381],[767,697]])
# Si el video esta vertical, se invierten las coordenadas de cada punto.
if orientation == 90:
    inverse = []

    for point in area_points:
        new_point = point[::-1]
        inverse.append(new_point)
    
    area_points = np.array(inverse)
    
frame_inicial = movementDetector(path_vid + vid_name, fps, area_points, kernel)
print('El frame inicial del video es: '+ str(frame_inicial))