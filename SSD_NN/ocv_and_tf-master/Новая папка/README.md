**1. Использование сети в OpenCV**
Ссылка на используемую сеть 
https://owncloud.k36.org/index.php/s/cY9VEQPtrNzc9cn

Неообходимо скачать архив и положить рядом с папкой проекта.
В файле "cut_video_demo_tensor.py" исправить пути инициализации у переменной "cvNet"

`cvNet = cv2.dnn.readNetFromTensorflow('../ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
                                      '../ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')`
                                      
В переменной media_src указан путь до медиа файла. Можно работать с видео, можно указать картинку.

**2. Использование любой сети Tensorflow**

**Ссылка на сеть обученную только на видео с тепловизора (Трактор)**
https://drive.google.com/open?id=1EGxf1EWjTpr_5SurdJM3sFKHZ1hQ46mM

**Ссылка на сеть (FLIR + трактор)**
https://drive.google.com/open?id=18Rf_tYh0py4MRhLzao9xkGjlpYoS4ENB

Неообходимо скачать архив и положить рядом с папкой проекта.
Скрипт с примером использования "tensorflow_without_Opencv.py". В нем нужно править все пути до сети и обрабатываемого видео файла

                                                      