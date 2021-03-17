from preprocess import grayscale, brightness, filter, morphology, threshold_process
import torch
import pandas
import PIL
import os
import cv2
import pandas as pd

class LemonDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform, param, test=False, timestamp=None):
        self.df = pandas.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.param = param
        self.test = test
        self.timestamp = timestamp
        self.filename_list = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.at[index, 'id']
        self.filename_list.append(filename)
        #img = PIL.Image.open(os.path.join(self.images_folder, filename))
        img = read_img(filename, self.param, self.test, self.timestamp)
        if self.transform is not None:
            img = self.transform(img)
        if not self.test:
            label = self.df.at[index, 'class_num']
            return img, label
        else:
            return img

def read_img(filename, param, test, timestamp):
    grayed = param['grayed']
    bright = param['bright']
    blur = param['blur']
    morph = param['morph']
    threshold = param['threshold']
    if test:
        label = "test"
    else:
        label = "train"
    img = cv2.imread("../dataset/" + label + "_images/" + filename)
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
    if timestamp:
        os.makedirs('./results/' + timestamp + '/preprocess', exist_ok=True)
        cv2.imwrite('./results/' + timestamp + '/preprocess/' + filename, img)
    return img

def write_img(img_list, filename, timestamp):
    os.mkdir('./results/' + timestamp + '/preprocess', exit=True)  # 結果を出力するディレクトリを作成
    for i in range(len(img_list)):
        cv2.imwrite('./results/' + timestamp + '/preprocess/' + filename[i], img_list[i])

def check_lemon_num():
    df = pd.read_csv('./dataset/train_images.csv',sep=',')
    df.head(3)

if __name__=='__main__':
    check_lemon_num()