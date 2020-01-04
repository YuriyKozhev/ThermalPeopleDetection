import cv2
import otsu

def full_method(img, L=256, M=64):
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )
    thresholds = otsu.modified_TSMO(hist, M=M, L=L)
    masks = otsu.multithreshold(img, thresholds)
    return thresholds, masks