import sys
import csv
#import cv2

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
    print(img)
    print(label)

def main():
    filename = "./dataset/train_images.csv"
    read_csv(filename)

if __name__ == '__main__':
    main()