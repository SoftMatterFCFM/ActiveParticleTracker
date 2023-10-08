"""
    Created on Monday May 29 2023 18:24 GMT-6
    Autor: Alberto Estudillo Moreno
    Last Modification: Thursday September 14 14:15 GMT-6
    
    Using autobbox V1.8, rastreo V1.10 and InitialFrame V0.14
"""

import cv2, time
import pandas as pd
import numpy as np
import exiftool as etl
import matplotlib.pyplot as plt

# ---------------- Funciones por modulo ---------------- #

#############################
#### Initial Frame v0.14 ####
#############################

class InitialFrame:  
  def lightIntensity(path: str, percent: float):
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
    
  def morphologicTransform(image, kernel):
    #Se aplican transformaciones morfologicas para mejorar la imagen binaria
    img_mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img_dil = cv2.dilate(img_mask, None, iterations= 1)
    demos = cv2.demosaicing(img_dil, code= cv2.COLOR_BayerBG2GRAY)
    bn = cv2.inRange(demos, np.array([25]), np.array([255]))
        
    return bn
    
  def movementDetector(path: str, fps: int, orientation: int,
                       area_points, kernel, opcion: int = 0):
    #Listas de apoyo
    if orientation == 90:
        inverse = []

        for point in area_points:
            new_point = point[::-1]
            inverse.append(new_point)
    
        area_points = np.array(inverse)
    
    frames = []
    frames_count = []
    contour_area = 0

    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, fps)
        
    while(capture.isOpened()):
        ret , frame = capture.read()
        fps = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            
        if ret == True:
            aux_img = InitialFrame.auxiliarImage(frame, area_points)
            img_mask = InitialFrame.morphologicTransform(aux_img, kernel)
            # cv2.imshow('mask', img_mask); cv2.waitKey(0)
 
            #Guarda el frame en una lista
            frames.append(img_mask)
            #Guarda el numero de frame que se esta guardando en la lista anterior
            frames_count.append(fps)
        
            #Superpone los frames guardados cuando la lista tiene un tamaño de 6
            if len(frames) == 5:
                sum_frames = sum(frames)
                #Encuentra el contorno del frame donde se sumaron los 6 frames
                contour, _ = cv2.findContours(sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                #Calcula el area del frame resultante de la suma
                for cont in contour:
                    contour_area = cv2.contourArea(cont)
                
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
                    print("Con area de: ", contour_area)
                    break
            
        else: 
            continue
    
    init_frame = frames_count[opcion]
    capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    ret , frame = capture.read()
    capture.release()   
    print('El frame inicial del video es: '+ str(init_frame))
    return init_frame, frame
    
################################
#### Auto Bounding Box v1.8 ####
################################

class BoundingBox:
  def get_auto_bbox(frame, delta: int, orientation: int, area_points, show: bool =False):  
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
    

#####################
#### Rastreo 1.10 ####
#####################

class rastreoCSRT:
  def get_rotation(video_path):
    ########## Se comprueba la orientacion del video #########
    with etl.ExifToolHelper() as et:
        metadata = pd.DataFrame(et.get_tags(video_path,tags="Rotation"))
    orientation = int(metadata['Composite:Rotation'][0])
    
    return orientation

  def traking_particle_CSRT(path: str, init_frame: int, end_frame: int,
                            orientation, bbox, irl = False):
      
    # Arreglo para guardar coordenadas
    coords = []
    
    ########## Colocar video en frame inicial #########
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    success, frame = capture.read()
    
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
          if fps == end_frame:
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
              cv2.putText(frame, 'Tracker', (400,20), font, 1, (0,255,0), 2)
              cv2.putText(frame, str(fps), (300,20), font, 1, (0,255,0), 2)
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
      
    plt.figure('rastreo')
    plt.plot(points[:,0],points[:,1])
    plt.show()
    return 0

# ---------------- Codigo Principal ---------------- #

####### Rutas de acceso para video y para guardado de datos ######
video_path = 'F:\\VParticles\\Puerta28\\'
video_name = '60-GH010283.MP4'
data_path = 'G:\\TrackingCompleto\\Resultados\\P28\\'
data_name = 'Tracking' + video_name[0:2] + '.dat'
full_path = video_path + video_name

####### Parametros #######
min_area = 1200
fps_final = 0
porcentaje = 0.97
delta = 4
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
area_points = np.array([[1201,697],[1201,381],[767,381],[767,697]])

####### Variables auxiliares ########
initial_time = time.time()
rotation = rastreoCSRT.get_rotation(full_path)
dark_fps = InitialFrame.lightIntensity(full_path, porcentaje)
fps_inicial, frame_inicial = InitialFrame.movementDetector(full_path, dark_fps, rotation, area_points, kernel, 4)

####### Rastreo de la particula #######
bbox = BoundingBox.get_auto_bbox(frame_inicial, delta, rotation, area_points)
tracking = rastreoCSRT.traking_particle_CSRT(full_path, fps_inicial, fps_final, rotation, bbox, True)

coordenadas = np.array(tracking)
np.savetxt(data_path+data_name, coordenadas)

final_time = time.time()
print('El tracking ' + video_name[0:2]+ ' tardo: %.2f'%((final_time-initial_time)/60)+' minutos.')
