# *Requisitos // Requirements*
- ExifTool  ->  https://pypi.org/project/PyExifTool/#getting-pyexiftool
- OpenCV 
- Numpy
- Pandas

# *Detalles // Details*
<details>
  <summary>Español // Spanish</summary>
  
# Información
En este repositorio hay carpetas con los modulos por separado (ingles o español), una llamada "Automatic",
el cual tiene un codigo que incluye todos los modulos para hacer el proceso de rastreo de manera automatizada.
Se recomienda usar el modulo de "RastreoV1.10" para rastrear la particula ya que, el codigo de "AutoTrack" esta 
en una fase preliminar.

Este codigo se apoya de la libreria [OpenCV](https://opencv.org/), en dado caso se requiera conocer mas a 
detalle las funciones utilizadas de la libreria, pueden recurrir a su [documentación](https://docs.opencv.org/).
Cada modulo mostrado aqui tiene una descripcion dentro del codigo como comentario.

# *Rastreo de una Partícula*
ActiveParticleTracker es un conjunto de codigos en Python con modulos para rastreo de particulas activas sinteticas en 2D.
## Modulo de Fotograma Inicial
Este modulo nos ayuda a encontrar el fotograma donde iniciara nuestro rastreo, primero se obtiene un fotograma
con baja luminosidad dependiendo del porcentaje de oscuridad deseado, luego se superponen los fotogramas para
detectar movimiento de la partícula y por ultimo, cuando se detecta movimiento, se utiliza uno de los fotogramas 
superpuestos* como nuestro fotograma inicial.

Versión mas reciente V0.14.

![Imagen2](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/ce9e5af8-471d-4804-a75d-126ae09385f1)  ======> 
![Imagen3](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/358e8cec-cd0b-4ff1-8163-b761698bcbfa)



*La superposición consta de 5 fotogramas sumados, si el fotograma inicial que
arroja el programa no funciona para iniciar el rastreo, puede tomar un numero
de fotograma ± 2 del numero de fotograma dado por el código.

## Modulo para Región de Interés.
Este modulo nos permite calcular las dimensiones y la posición del área donde esta la partícula,
para evitar el ruido que pueda haber en las imágenes del video. Esta región de interés son necesarias
para usar el método DCF-CSRT de rastreo. 

Versión mas reciente V1.8.

![Imagen1](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/35b84123-c825-4963-9021-13ddae15857a)


## Modulo de Rastreo.
Se usa el modulo de rastreo para obtener la trayectoria de una partícula activa sintética.
Este modulo utiliza como base el método DCF-CSRT *(Discriminative Correlation Filter with Channel 
and Spatial Reliability Tracker)* [^2] . 

Versión mas reciente V1.10.

![Screenshot 2023-10-04 152530](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/262fd5c7-eabc-464a-9aaa-3e08843f14d6)


[^2]: Lukežic, A., Vojír, T., Zajc, L. and Kristan, M. (2017). Discriminative Correlation Filter Tracker with Channel and Spatial Reliability, International Journal of Computer Vision. https://doi.org/10.1007/s11263-017-1061-3.
</details>

<details>
  <summary>Inglés // English</summary>

# *Information*
In this repository there are folders with separate modules (English or Spanish) one named "Automatic" which has
a script that includes all the modules to do the tracking process in an automated way. We recommend the use of
"TrackerV1.10" module to track the particle since the "AutoTrack" script is in a preliminary phase.

These codes are supported by the [OpenCV](https://opencv.org/) library, if you need to know more details about the functions
used by the library, you can refer to its [documentation](https://docs.opencv.org/). Each module shown here has a description inside the script as a comment.


# *Active Particle Tracking*
ActiveParticleTracker is a set of Python codes with modules for tracking synthetic active particles in 2D.
## Initial Frame Module 
This module helps us to find the frame in which the tracking is going to start, it obtains a frame with low luminosity according to a darkness percentage, then it uses 
the frame superposition to detect the particle's motion and finally, when it detects motion, we use one of the superpositioned frames* to be our initial frame. 

Latest version V0.14.

![Imagen2](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/ce9e5af8-471d-4804-a75d-126ae09385f1)  ======> 
![Imagen3](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/358e8cec-cd0b-4ff1-8163-b761698bcbfa)

*The superposition method consists of 5 frames added together, if the returned initial frame of the module
doesn't work to initialize the tracking, you can use a frame either two places ahead or behind the frame returned by
the module $(fr - 2 < fr < fr + 2)\text{, where }fr = \text{Frame returned}$


## Region of Interest Module
This module allows us to compute the dimension and position of the area where the particle is located to avoid
the noise in the image. This region of interest is needed to use the DCF-CSRT tracking method.

Latest version V1.8.

![Imagen1](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/35b84123-c825-4963-9021-13ddae15857a)

## Tracking Module
The module is used to obtain the trajectory of the synthetic active particle. This module uses the DCF-CSRT method
as a base. (Discriminative Correlation Filter with Channel and Spatial Reliability Tracker)[^2]. 

Latest version V1.10.

![Screenshot 2023-10-04 152530](https://github.com/SoftMatterFCFM/ActiveParticleTracker/assets/147351815/262fd5c7-eabc-464a-9aaa-3e08843f14d6)

[^2]: Lukežic, A., Vojír, T., Zajc, L. and Kristan, M. (2017). Discriminative Correlation Filter Tracker with Channel and Spatial Reliability, International Journal of Computer Vision. https://doi.org/10.1007/s11263-017-1061-3.
</details>
