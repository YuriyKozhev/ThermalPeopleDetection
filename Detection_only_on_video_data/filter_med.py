import cv2
import matplotlib.pyplot as plt
import numpy as np
from auto_tsmo import full_method

def mFilter(thresholds, masks, relative = False):
    if relative:
        thresh = ((masks[int(len(thresholds)//1.437)]*255)).astype(np.uint8)
    else:
        thresh = ((masks[16]*255)).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8) 
    thresh = cv2.dilate(thresh, kernel, iterations=10) 
      
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def mPostProcessing(contours):
    min_height = 100
    min_width = 65
    ratio = 1.2
                
    rects = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        
        if w > min_width  and h > min_height and h/w > ratio:
            rects.append((x,y,w,h))
    
    return rects