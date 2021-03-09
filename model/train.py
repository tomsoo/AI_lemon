from bi import LemonDataset
from model import CNN
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import datetime
import os

def main():
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
        'epoch_num' : 10,
    })  # 追加パラメータ

    # 実行時のパラメータをファイルとして記録
    with open(f'{result_dir}parameter.json', mode='w') as f:
        json.dump(param, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    # パラメータの読み込み
    train_file = param['train_file']
    train_folder = param['train_folder']

    valid_file = param['valid_file']
    valid_folder = param['valid_folder']

    batch_size = param['batch_size']

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (0.5,))]) # 画像の読み込み
    print("Loading train image...")
    train_dataset = LemonDataset(train_file, train_folder, trans, param)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print("Loading valid image...")
    valid_dataset = LemonDataset(valid_file, valid_folder, trans, param)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)


    # 学習
    epoch_num = param['epoch_num']

    # ネットワークのインスタンス作成
    cnn = CNN()

    # 損失関数とオプティマイザーの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

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

            # パラメータを0にリセット
            optimizer.zero_grad()

            # 順方向の計算、損失計算、バックプロパゲーション、パラメータ更新
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 計算状態の出力
            running_loss += loss.item()
        print('epoch: %d, loss: %.5f' % (epoch + 1, running_loss))

        valid_runnning_loss = 0.0
        with torch.no_grad():
            for data in valid_dataloader:
                inputs, labels = data

                outputs = cnn(inputs)

                valid_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs,1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

                valid_acc = valid_correct/valid_total * 100

                valid_loss_list.append(valid_loss)
                valid_acc_list.append(valid_acc)
        print('valid loss:{}, valid acc:{}'.format(valid_loss, valid_acc))


    print("Finished Training")

    # 計算結果のモデルを保存
    torch.save(cnn.state_dict(), './results/' + timestamp + '/cifar_net.pth')

    print("Finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='../dataset/train_images.csv')
    parser.add_argument('--train_folder', type=str, default='../dataset/train_images/')
    parser.add_argument('--valid_file', type=str, default='../dataset/eval_images.csv')
    parser.add_argument('--valid_folder', type=str, default='../dataset/train_images/')
    args = parser.parse_args()  # 引数解析
    main()