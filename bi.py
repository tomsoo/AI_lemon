from preprocess import grayscale, brightness, filter, morphology, threshold_process
import torch
import pandas
import csv
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

class LemonDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform, param, grayed, test=False):
        self.df = pandas.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.param = param
        self.test = test
        self.grayed = grayed
        self.filename_list = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.at[index, 'id']
        self.filename_list.append(filename)
        #img = read_img(filename, self.param, self.test)
        if self.grayed:
            img = cv2.imread(self.images_folder + filename, 0)
        else:
            img = cv2.imread(self.images_folder + filename)
        if self.transform is not None:
            img = self.transform(img)
        if not self.test:
            label = self.df.at[index, 'class_num']
            return img, label
        else:
            return img

def read_img(filename, param, test):
    grayed = param['grayed']
    bright = param['bright']
    blur = param['blur']
    morph = param['morph']
    threshold = param['threshold']
    if test:
        label = "test"
    else:
        label = "train"
    img = cv2.imread("./dataset/" + label + "_images/" + filename)
    if grayed:
        img = grayscale(img)
    if bright:
        img = brightness(img)
    if blur:
        img = filter(img)
    if morph:
        img = morphology(img)
    if threshold:
        img = threshold_process(img)
    os.makedirs('./dataset/' + label + '_preprocess', exist_ok=True)
    cv2.imwrite('./dataset/' + label + '_preprocess/' + filename, img)
    return img

def write_img(img_list, filename, timestamp):
    os.mkdir('./results/' + timestamp + '/preprocess', exit=True)  # 結果を出力するディレクトリを作成
    for i in range(len(img_list)):
        cv2.imwrite('./results/' + timestamp + '/preprocess/' + filename[i], img_list[i])

def check_lemon_num():
    df = pd.read_csv('./dataset/train_images.csv',sep=',')
    df.head(3)

def loss_visualize(name, loss, num, timestamp, k):
    plt.figure()
    loss_list = []
    for l in loss:
        loss_list.append(l.to('cpu').detach().numpy().copy())
    plt.plot([x for x in range(num)], loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("./results/" + timestamp + "/" + name + "_" + str(k) + ".png")
    plt.close()

def get_train_data(filename, param):
    csv_file = open(filename)
    f = next(csv.reader(csv_file))
    f = csv.reader(csv_file, delimiter=",")
    for line in f:
        read_img(line[0], param, False)

def get_test_data(filename, param):
    csv_file = open(filename)
    f = next(csv.reader(csv_file))
    f = csv.reader(csv_file, delimiter=",")
    for line in f:
        read_img(line[0], param, True)


if __name__=='__main__':
    check_lemon_num()
