from bi import LemonDataset
from model import CNN
import csv
import torch
import torchvision
import argparse

def main():
    # 実行時パラメータ群
    param = vars(args)  # コマンドライン引数を取り込み
    param.update({
        # 前処理
        'grayed': True,  # グレースケール
        'bright': False,  # 輝度調整
        'blur': True,  # 平滑化(フィルター)
        'morph': False,  # 平滑化(モルフォロジー)
        'threshold': False,  # 閾値処理
        # 学習
        'batch_size': 4,
    })  # 追加パラメータ

    # パラメータの読み込み
    eval_file = param['eval_file']
    eval_folder = param['eval_folder']
    test_file = param['test_file']
    test_folder = param['test_folder']
    model_path = param['model_path']
    batch_size = param['batch_size']
    grayed = param['grayed']

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (0.5,))]) # 画像の読み込
    '''
    print("Loading eval image...")
    eval_dataset = LemonDataset(eval_file, eval_folder, trans, param, grayed)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True)
    '''

    print("Loading test image...")
    test_dataset = LemonDataset(test_file, test_folder, trans, param, grayed, True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # ネットワークモデル読み込み
    cnn = CNN(grayed)
    cnn.load_state_dict(torch.load(model_path))

    # 精度の計算
    '''
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_dataloader:
            images, labels = data
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d test images: %.3f %%' % (total, 100 * correct / total))
    '''

    # テスト結果の出力
    predicted_list = []
    with torch.no_grad():
        for images in test_dataloader:
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(predicted)):
                predicted_list.append(int(predicted[i]))

    with open('./results/results.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(len(predicted_list)):
            writer.writerow([test_dataset.filename_list[i], predicted_list[i]])

    print("Finished testing")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default='../dataset/eval_images.csv')
    parser.add_argument('--eval_folder', type=str, default='../dataset/train_images/')
    parser.add_argument('--test_file', type=str, default='../dataset/test_images.csv')
    parser.add_argument('--test_folder', type=str, default='../dataset/test_images/')
    parser.add_argument('--model_path', type=str, default='./results/20210321-032803/cifar_net.pth')
    args = parser.parse_args()  # 引数解析
    main()