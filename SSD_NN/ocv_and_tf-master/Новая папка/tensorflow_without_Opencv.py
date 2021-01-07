# Example of using a neural Tensorflow network without OpenCV.
# This code is compatible with networks built on Tensorflow version 1.14, 1.15

import numpy as np
import os
import sys
from FpsCounter import FpsCounter

# if use tf version >=2+ uncomment next string
#import tensorflow.compat.v1 as tf

# if use tf version < 2.0
import tensorflow as tf

import cv2

#media_src = 'samples/project.avi'
media_src = 'samples/sam1.mp4'
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(media_src)
fps = FpsCounter()

sys.path.append("..")

PATH_TO_CKPT = '../ssd_mobilenet_v2_coco_2020_02_02/frozen_inference_graph.pb'
#PATH_TO_CKPT = '../ssd_mobilenet_v2_coco_2020_01_29_3/frozen_inference_graph.pb'
#PATH_TO_CKPT = '../ssd_mobilenet_v2_coco_2020_01_29_2/frozen_inference_graph.pb'
#PATH_TO_CKPT = '../ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

def add_fps(clone_frame):
    global fps
    fps.checkpoint()
    cv2.putText(clone_frame, str(fps), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            fps.start()
            ret, frame = cap.read()
            if ret == False:
                break
            rows = frame.shape[0]
            cols = frame.shape[1]
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(frame, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Using Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            print("detected = "+str(num_detections[0]))
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            # Visualize only detected boxes
            if (num_detections>0):
                for i in range(0,int(num_detections[0])):
                    # Check scores for visualise
                    if scores[i] < 0.2 :
                        continue
                    detection = boxes[i]
                    print(str(detection) +' score='+ str(scores[i] ) + ' cols='+str(cols) + ' rows='+str(rows))
                    left = detection[1] * cols
                    top = detection[0] * rows
                    right = detection[3] * cols
                    bottom = detection[2] * rows
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

            #cv2.imshow('object detection', cv2.resize(frame, (800,600)))

            add_fps(frame)
            cv2.imshow('object detection', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
cv2.destroyAllWindows()