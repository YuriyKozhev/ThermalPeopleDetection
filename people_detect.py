#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import cv2
import otsu
import matplotlib.pyplot as plt
import numpy as np
import glob
import math
from PlotToImage import PlotToImage
from auto_tsmo import full_method
import time
import json
from sklearn.svm import LinearSVC
from joblib import dump, load
from feature_extraction import getFeaturesVector
from feature_extraction import getFeatures


# In[2]:


def ImShow(im, size=(14,17)):
    plt.figure(figsize = size)
    plt.imshow(im, 'gray')
    plt.show()

def HistPlot(vals, size=(14,17)):
    plt.figure(figsize = size)
    plt.plot(np.arange(len(vals)), vals)
    plt.show()


# In[3]:


def symm(array, perc = .11):
    array /= np.max(array)
    
    pairs = []
    for i in range(len(array)):
        for j in range(i, min(i + len(array)//3, len(array))):
            if abs(array[i]-array[j]) < perc and (array[i] != 0 and array[j] != 0):
                pairs.append([i,j])
    sym = pairs
    
    new_dens = np.ones(len(array))

    for i in range(len(sym)):
        if (sym[i][0] + sym[i][1]) % 2 == 0:
            k = int(((sym[i][1] - sym[i][0])/2) + sym[i][0])
    #         print(k)
            new_dens[k] *= 1 + (array[k]/max(array))
        else:
            k = int((((sym[i][1] - sym[i][0])/2) - .5) + sym[i][0])
    #         print(k)
            new_dens[k] *= 1 + (array[k]/max(array))
            new_dens[k+1] *= 1 + (array[k]/max(array))
    
    return new_dens > 1


# In[4]:


def whiteDensity(im, n = 1, val = 254):
    new_im = cv2.threshold(im, val, 255, cv2.THRESH_BINARY)[1]
    den = [np.sum([np.sum([new_im[j,i] for j in range(im.shape[0])]) for i in range(k, k+n)]) for k in range(0, im.shape[1], n)]
    den /= np.max(den)
    return den


# In[5]:


def grayDensity(im, n = 1):
    new_im = im
    den = [np.sum([np.sum([new_im[j,i] for j in range(im.shape[0])]) for i in range(k, k+n)]) for k in range(0, im.shape[1], n)]
    den /= np.max(den)
    return den


# In[6]:


def isCorrect(reg):
    den = grayDensity(reg)
    min_val = den.min()
    int_val = reg.sum() / (reg.shape[0] * reg.shape[1])
  #  print(int_val)
    if min_val < 0.75:
        if int_val > 80:
            return True
        
    return False


# In[7]:


# from scipy.ndimage.filters import gaussian_filter1d as gauss
# def getImMask(im):
#     return gauss(whiteDensity(im, val = thresh_val), 5) > 0.01


# In[8]:


def getRegionsAndMasks(im, im_mask, return_regions = True):
    reg_masks = []
    mask = []
    for i, s in enumerate(im_mask):
        if s:
            mask.extend([i])
        else:
            if len(mask) > 5:
                reg_masks.append(mask)
            mask = []
    if len(mask) > 5:
        reg_masks.append(mask)
    if not return_regions:
        return reg_masks
    regions = [np.array([[im[i,j] for j in mask] for i in range(im.shape[0])], dtype=np.uint8) for mask in reg_masks]
    return regions, reg_masks


# In[9]:


FLIR_model = load('FLIR_classifier')
video_model = load('video_classifier')

# In[10]:


def getRects(regions, reg_masks):
    filtered_cnts = []
    for i, reg in enumerate(regions):
        thresholds, masks = full_method(reg, L = 2**8, M=128)
        thresh = (((masks[-2]+masks[-1])*255)).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8) 
        thresh = cv2.dilate(thresh, kernel, iterations=10) 
        
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if not (w > 10 and h > 20 and h / w > 0.5 and h / w < 3):
                continue
            if not isCorrect(reg[y:y+h,x:x+w]):
                continue
            x += reg_masks[i][0]
            filtered_cnts.append([x,y,w,h])
            
    return filtered_cnts 

def filterRects(im_path, rects, video_or_FLIR):
    '''
    true_rects = []
    for rect in rects:
        x,y,w,h = rect
        img = frame[y:y+h,x:x+w]
        preds = net_predict_image(img)
        if len(preds) != 0:
            true_rects.append(preds)
    return true_rects
    '''
    if video_or_FLIR == 'video':
        model = video_model
    elif video_or_FLIR == 'FLIR':
        model = FLIR_model
    else:
        model = None
    true_rects = []
    for rect in rects:
        if bool(model.predict(getFeatures((im_path, rect), True, True))):
            true_rects.append(rect)
    return true_rects


# In[11]:


def drawContours(frame, rects):
    for rect in rects:
        x,y,w,h = rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        
def drawRects(frame, rects):
    for rect in rects:
        x,y,w,h = rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

def saveRects(rects, im_path, predictions):
    for rect in rects:
        if im_path in predictions:
            predictions[im_path].append(rect)
        else:
            predictions[im_path] = []
            predictions[im_path].append(rect)


# In[21]:


def main(im_path, result):
 #  print('main')
    frame = cv2.imread(im_path)
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    thresholds, masks = full_method(im, L = 2**8, M=128)
    N = 8
    thresh_val = thresholds[-1]
    den = whiteDensity(im, n = N, val = thresh_val)  

    sym = symm(den, 0.1)
    sym = np.array([[sym[i] for j in range(N)] for i in range(len(sym))]).reshape(-1,) 
    sym_im = np.array([[im[i,j] for j in range(im.shape[1])]*sym for i in range(im.shape[0])], dtype = np.uint8)
    
    regions, reg_masks = getRegionsAndMasks(im, sym)
    rects = getRects(regions, reg_masks)
    rects = filterRects(im_path, rects, 'video')
    drawRects(frame, rects)

    if result == 'IMAGE' or result == 'COUNT_IMG':
        cv2.imwrite('result.jpeg', frame)
    if result == 'CONTOURS' or result == 'COUNT_IMG':
         print(*rects)


# In[24]:


# im_path = r'C:\Users\Yuriy\Google\VideoData\Frames\video_frame_1305.jpeg'
# main(im_path)


# In[26]:
#print ('a')

import sys

if __name__ == "__main__":
    #im_path = sys.argv[1]
    #main(im_path)
    args = {'mode': None,
            'dataset': None,
            'json': None,
            'path': None,
            'return': None}
    i = 1
    while True:
        try:
            arg = sys.argv[i]
            if arg == '--mode':
                args['mode'] = sys.argv[i+1]
            if arg == '--dataset':
                args['dataset'] = sys.argv[i + 1]
            if arg == '--json':
                args['json'] = sys.argv[i+1]
            if arg == '--path':
                args['path'] = sys.argv[i+1]
            if arg == '--return':
                args['return'] = sys.argv[i+1]
        except:
            break
        i += 1
  #  print(*args.items())
    if args['mode'] == 'PREDICT':
        main(args['path'], args['return'])

 #   print('success')


# In[ ]:




