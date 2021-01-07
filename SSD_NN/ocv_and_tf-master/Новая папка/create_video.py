# This file is a part of test_ocv project.
#
# Copyright (C) 2019, HSE MIEM, all rights reserved.
# Third party copyrights are property of their respective owners.
#
# Use this script to a create video file from many pictures


import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('E:/FLIR_ADAS_1_3/FLIR_ADAS_1_3/FLIR_ADAS_1_3/val/thermal_8_bit/*.jpeg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()