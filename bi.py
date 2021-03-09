from preprocess import grayscale, brightness, filter, morphology, threshold_process
import torch
import pandas
import PIL
import os
import cv2
import pandas as pd

class LemonDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform, param):
        self.df = pandas.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.param = param

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.at[index, 'id']
        label = self.df.at[index, 'class_num']
        #img = PIL.Image.open(os.path.join(self.images_folder, filename))
        img = read_img(filename, self.param)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def read_img(filename, param):
    grayed = param['grayed']
    bright = param['bright']
    blur = param['blur']
    morph = param['morph']
    threshold = param['threshold']
    img = cv2.imread("../dataset/train_images/" + filename)
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
    return img

def write_img(img_list, filename, timestamp):
    os.mkdir('./results/' + timestamp + '/preprocess')  # 結果を出力するディレクトリを作成
    for i in range(len(img_list)):
        cv2.imwrite('./results/' + timestamp + '/preprocess/' + filename[i], img_list[i])

def check_lemon_num():
    df = pd.read_csv('./dataset/train_images.csv',sep=',')
    df.head(3)

if __name__=='__main__':
    check_lemon_num()