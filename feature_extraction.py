from skimage.feature import local_binary_pattern
from skimage.feature import hog
import numpy as np
import cv2
from HistogramOfOpticalFlow.py import hof

def getHOOF(im_path, cnt):
    x,y,w,h = cnt
    img = cv2.imread(im_path, 0)[y:y+h,x:x+w]
    
    width = 64
    height = 128
    
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    
    fd = hof(resized, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), feature_vector = True)
    
    return fd

def getHOG(im_path, cnt):
    x,y,w,h = cnt
    img = cv2.imread(im_path, 0)[y:y+h,x:x+w]
    
    width = 64
    height = 128
    
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    
    fd = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), feature_vector = True)
    
    return fd

def getLBP(im_path, cnt):
    x,y,w,h = cnt
    img = cv2.imread(im_path, 0)[y:y+h,x:x+w]
    
    width = 64
    height = 128
    step_width = 8
    step_height = 16
    
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
    
    METHOD = 'nri_uniform'
    radius = 1
    n_points = 8 * radius
    
    hists = []

    for i in range(0, width, step_width):
        for j in range(0, height, step_height):
            lbp = local_binary_pattern(resized[j:j+step_width,i:i+step_height], n_points, radius, METHOD)
         #   print(lbp.max(), lbp.min())
            hist, bins = np.histogram(lbp, np.arange(58+2), density =True)
          #  print(bins)
            hists.extend(hist)
    return np.array(hists)

def getFeatures(sample, add_HOG, add_LBP):
    #print('s', sample0, sample1)
    if add_HOG:
        if add_LBP:
    #        print(sample[0],sample[1])
            return np.append(getHOG(sample[0],sample[1]), getLBP(sample[0],sample[1])).reshape(1,-1)
        else:
            return getHOG(sample[0],sample[1]).reshape(1,-1)
    else:
        return getLBP(sample[0],sample[1]).reshape(1,-1)

def getFeaturesVector(data, add_HOG, add_LBP):
    features = []
    for sample in data:
        features.append(getFeatures(sample, add_HOG, add_LBP).reshape(-1,))
    return np.array(features)