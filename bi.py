from preprocess import grayscale, threshold_process
import os
import csv
import cv2

def read_csv(filename):
    img = []
    label_str = []
    csv_file = open(filename)
    f = next(csv.reader(csv_file))
    f = csv.reader(csv_file, delimiter=",")
    for line in f:
        img.append(line[0])
        label_str.append(line[1])
    label = [int(i) for i in label_str]
    return img, label

def read_img(filename, data, param):
    img = []
    grayed = param['grayed']
    threshold = param['threshold']
    for f in filename:
        img.append(cv2.imread("../dataset/" + data + "_images/" + f))
    if grayed:
        img = grayscale(img)
    if threshold:
        img = threshold_process(img)
    return img

def write_img(img_list, filename, timestamp):
    os.mkdir('./results/' + timestamp + '/preprocess')  # 結果を出力するディレクトリを作成
    for i in range(len(img_list)):
        cv2.imwrite('./results/' + timestamp + '/preprocess/' + filename[i], img_list[i])