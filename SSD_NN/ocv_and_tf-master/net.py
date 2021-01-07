import numpy as np
import os
import sys
import tensorflow as tf
import cv2

#sys.path.append("..")

def net_predict(img, source):
    path = r'C:\Users\Yuriy\Net2//'
    
    if source == 'video':
        PATH_TO_CKPT = path + '/ssd_mobilenet_v2_coco_2020_02_02/frozen_inference_graph.pb'
    elif source == 'FLIR':
        PATH_TO_CKPT = path + '/ssd_mobilenet_v2_coco_2020_02_17/frozen_inference_graph.pb'
    else:
        assert(False)
        
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            frame = img
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
            print('num_detections =', num_detections)
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
            cv2.imshow('object detection', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()        
    