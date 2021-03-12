import cv2
import numpy as np

def grayscale(img):
    #print("Grayscale...")
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed

def brightness(img):
    #print("Adjusting the brightness...")
    bright = (img - np.mean(img)) / np.std(img) * 32 + 64
    return bright

def filter(img):
    #print("Smoothed with a filter...")
    filtered = cv2.GaussianBlur(img, (11, 11), 0)
    return filtered

def morphology(img_list):
    opened = []
    #print("Smoothed with morphology...")
    kernel = np.ones((3, 3), np.uint8)
    for img in img_list:
        opened.append(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2))
    return opened

def threshold_process(img):
    #print("Threshold processing...")
    under_thresh = 65
    upper_thresh = 200
    maxValue = 255
    th, drop_back = cv2.threshold(img, under_thresh, maxValue, cv2.THRESH_BINARY) # 背景を落とす
    th, clarify_born = cv2.threshold(img, upper_thresh, maxValue, cv2.THRESH_BINARY_INV) # 境界の明確化
    threshold = np.minimum(drop_back, clarify_born)
    return threshold