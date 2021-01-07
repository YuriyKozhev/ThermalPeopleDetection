# This file is a part of test_ocv project.
#
# Copyright (C) 2019, HSE MIEM, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Use this script to a create image files and csv file with ROI coordinates of human from video file
# Use a mouse for the select ROI
# Use key "p" - pause and play,
# key "q" - for exit app,
# key "s" - save picture to the file and save coordinates,
# key "w" - for the clear selected ROI on frame.

from builtins import print
from Poi import Poi
import cv2, datetime
import numpy as np

frame = ""
isShowRect = False
cropping = False
isPause = False
outfolder = "outs/"
now = datetime.datetime.now()
now_str = now.strftime("%d-%m-%y_%H-%M_")
file_points = open(outfolder+"pois_file_"+now_str+".csv","w")
file_points.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
curnumber = 0



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


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('samples/sam1.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_and_crop)
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    if not isPause:
        ret, frame = cap.read()

    if ret == True:
        clone_frame = frame.copy()
        add_rect()
        # Display the resulting frame
        cv2.imshow('Frame', clone_frame)
        last_key = cv2.waitKey(25) & 0xFF
        # Press S on keyboard to save regiot to file
        if last_key == ord('s'):
            if len(refPt) == 2:
                imgFileName = now_str + str(curnumber) + ".jpg"
                height, width = frame.shape[:2]
                poi = Poi(imgFileName, width, height, "myPerson", refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1])
                print(poi)
                file_points.write(poi.__str__()+"\n")
                roi = frame[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv2.imwrite(outfolder+imgFileName, frame)
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


    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
file_points.close()
cv2.destroyAllWindows()
