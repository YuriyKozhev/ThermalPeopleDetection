# This file is a part of test_ocv project.
#
# Copyright (C) 2019, HSE MIEM, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# This script demonstrates the possibilities of using a neural network to detect people in a video stream.
# Use a mouse for the select a ROI
# Use key "p" - pause and play,
# key "q" - for exit app,
# key "s" - save picture to the file and save coordinates,
# key "w" - for the clear selected ROIs on frame.

from builtins import print
import cv2, datetime
from FpsCounter import FpsCounter

frame = ""
clone_frame = ""
isShowRect = False
cropping = False
isPause = False
out_folder: str = "outs/"
now = datetime.datetime.now()
now_str = now.strftime("%d-%m-%y_%H-%M_")
print("Before net loader...")
# cvNet = cv2.dnn.readNetFromTensorflow('../ssd_mobilenet_v2_coco_2020_01_29_2/frozen_inference_graph.pb',
#                                       '../ssd_mobilenet_v2_coco_2020_01_29_2/graph.pbtxt'
#                                       )

cvNet = cv2.dnn.readNetFromTensorflow('../ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
                                      '../ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
                                      )
print("Net loaded!!!!")
#media_src = 'samples/Flir_video.avi'
#media_src = 'samples/project.avi'
media_src = 'samples/sam1.mp4'

# cvNet = cv2.dnn.readNetFromTensorflow('../ssd_coco_person_inference_graph_1760/frozen_inference_graph.pb','../ssd_coco_person_inference_graph_1760/graph.pbtxt')
# print("Before net loader...")
# cvNet = cv2.dnn.readNetFromTensorflow('../ssd_coco_person_inference_graph_25.11.19_2/frozen_inference_graph.pb','../ssd_coco_person_inference_graph_25.11.19_2/graph.pbtxt')
#print("Net loaded!!!!")
curnumber = 0

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(media_src)
fps = FpsCounter()

# Apply the neural network to the frame and encircles the area where the person is found.
# Returns a frame with rectangles
def apply_net(appFrame: object, cvNet: object, percent: float) -> object:
    # These are the correct operating parameters.
    cvNet.setInput(cv2.dnn.blobFromImage(appFrame, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        classID = int(detection[1])
        # score - probability, сlassID == 1 check only persons
        if score > percent and classID == 1:
            # print(detection)
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(appFrame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

    return appFrame



def add_fps(clone_frame):
    global fps
    fps.checkpoint()
    cv2.putText(clone_frame, str(fps), (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)


# draw a rectangle around the region of interest
def add_rect():
    global clone_frame
    try:
        if isShowRect:
            cv2.rectangle(clone_frame, refPt[0], refPt[1], (0, 255, 0), 2)

        if cropping:
            cv2.rectangle(clone_frame, refPt[0], movePoint, (0, 0, 255), 2)
    except Exception as e:
        print("Frame error! " + str(e.__class__))


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, isShowRect, movePoint

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            movePoint = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        isShowRect = False

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        isShowRect = True


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
    exit()

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_and_crop)
# Read until video is completed
last_key = 0
while (cap.isOpened()):
    if last_key == ord('q'):
        break

    fps.start()
    # Capture frame-by-frame
    if not isPause:
        ret, frame = cap.read()

    if ret == True:
        clone_frame = frame.copy()
        rows = frame.shape[0]
        cols = frame.shape[1]
        # Parameters from examples in documentations, but it does not work!!!!
        # cvNet.setInput(cv2.dnn.blobFromImage(cloneframe, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))

        clone_frame = apply_net(clone_frame, cvNet, 0.3)
        add_rect()
        add_fps(clone_frame)
        # Display the resulting frame
        cv2.imshow('Frame', clone_frame)
        last_key = cv2.waitKey(25) & 0xFF
        # Press S on keyboard to save regiot to file
        if last_key == ord('s'):
            if len(refPt) == 2:
                roi = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv2.imwrite(out_folder + now_str + str(curnumber) + '.jpg', roi)
                curnumber += 1
        # Press P on keyboard to PAUSE or PLAY video
        if last_key == ord('p'):
            isPause = not isPause
        # Press W to Clear region of interest
        if last_key == ord('w'):
            isShowRect = False
        # Press Q on keyboard to  exit
        elif last_key == ord('q'):
            break
    # Break the loop reads frames
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
