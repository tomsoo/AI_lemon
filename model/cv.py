from bi import LemonDataset, loss_visualize
from model import CNN
from guppy import hpy
from sklearn.model_selection import KFold
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
import argparse
import json
import datetime
import os


#@profile
def main():
    #h = hpy()
    timestamp = "{0:%Y%m%d-%H%M%S}".format(datetime.datetime.now())  # タイムスタンプ
    result_dir = './results/' + timestamp + '/'  # 結果を出力するディレクトリ名
    os.mkdir(result_dir)  # 結果を出力するディレクトリを作成

    # 実行時パラメータ群
    param = vars(args)  # コマンドライン引数を取り込み
    param.update({
        # 前処理
        'grayed': False, # グレースケール
        'bright': False, # 輝度調整
        'blur' : False, # 平滑化(フィルター)
        'morph' : False, # 平滑化(モルフォロジー)
        'threshold' : False, # 閾値処理
        # 学習
        'batch_size' : 4,
        'epoch_num' : 100,
    })  # 追加パラメータ

    # 実行時のパラメータをファイルとして記録
    with open(f'{result_dir}parameter.json', mode='w') as f:
        json.dump(param, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    # パラメータの読み込み
    data_file = param['data_file']
    data_folder = param['data_folder']

    batch_size = param['batch_size']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device is {}".format(device))

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (0.5,))]) # 画像の読み込み
    # kfold
    dataset = LemonDataset(data_file, data_folder, trans, param, timestamp=timestamp)
    kf = KFold(n_splits=5)
    k = 0
    cv = 0

    # 学習
    epoch_num = param['epoch_num']

    # ネットワークのインスタンス作成
    grayed = param['grayed']
    cnn = CNN(grayed)
    cnn = cnn.to(device)

    # 損失関数とオプティマイザーの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

    for _fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
        print("k = " + str(k))
        print("Loading train image...")
        train_dataset = Subset(dataset, train_index)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        print("Loading valid image...")
        valid_dataset = Subset(dataset, valid_index)
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False)

        train_total = 0
        train_correct = 0
        train_loss_list = []
        train_acc_list = []

        valid_total = 0
        valid_correct = 0
        valid_loss_list = []
        valid_acc_list = []

        # 学習ループ
        for epoch in range(epoch_num):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                # データをリストに格納
                inputs, labels = data
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)
                # パラメータを0にリセット
                optimizer.zero_grad()

                # 順方向の計算、損失計算、バックプロパゲーション、パラメータ更新
                outputs = cnn(inputs)
                _, predicted = torch.max(outputs,1)
                train_loss = criterion(outputs, labels)
                train_loss.backward()

                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                train_acc = train_correct/train_total*100
                optimizer.step()

                # 計算状態の出力
                running_loss += train_loss.item()
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('epochs: {}'.format(epoch + 1))
            print('train loss:{}, train acc:{}'.format(train_loss, train_acc))
            #print(h.heap())

            valid_runnning_loss = 0.0
            with torch.no_grad():
                for data in valid_dataloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

                    outputs = cnn(inputs)

                    valid_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs,1)
                    valid_total += labels.size(0)
                    valid_correct += (predicted == labels).sum().item()

                    valid_acc = valid_correct/valid_total * 100
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)

            print('valid loss:{}, valid acc:{}'.format(valid_loss, valid_acc))
        loss_visualize("train", train_loss_list, epoch_num, timestamp, k)
        loss_visualize("valid", valid_loss_list, epoch_num, timestamp, k)
        print("Finished Training")

        k += 1
        cv += valid_loss / kf.n_splits

    # 計算結果のモデルを保存
    torch.save(cnn.state_dict(), './results/' + timestamp + '/cifar_net.pth')
    print("Finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='../dataset/all_images.csv')
    parser.add_argument('--data_folder', type=str, default='../dataset/train_images/')
    args = parser.parse_args()  # 引数解析
    main()