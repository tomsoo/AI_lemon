import cv2
import numpy as np

def grayscale(img_list):
    grayed = []
    for img in img_list:
        grayed.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return grayed

def threshold_process(img_list):
    threshold = []
    for img in img_list:
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        under_thresh = 105
        upper_thresh = 145
        maxValue = 255
        th, drop_back = cv2.threshold(grayed, under_thresh, maxValue, cv2.THRESH_BINARY) # 背景を落とす
        th, clarify_born = cv2.threshold(grayed, upper_thresh, maxValue, cv2.THRESH_BINARY_INV) # 境界の明確化
        threshold.append(np.minimum(drop_back, clarify_born))
    return threshold