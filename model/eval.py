from bi import LemonDataset
from model import CNN
import torch
import torchvision
import argparse

def main():
    # 実行時パラメータ群
    param = vars(args)  # コマンドライン引数を取り込み
    param.update({
        # 前処理
        'grayed': True,  # グレースケール
        'bright': True,  # 輝度調整
        'blur': True,  # 平滑化(フィルター)
        'morph': False,  # 平滑化(モルフォロジー)
        'threshold': True,  # 閾値処理
        # 学習
        'batch_size': 4,
        'epoch_num': 10,
    })  # 追加パラメータ

    # パラメータの読み込み
    eval_file = param['eval_file']
    eval_folder = param['eval_folder']
    model_path = param['model_path']
    batch_size = param['batch_size']

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5,), (0.5,))]) # 画像の読み込み
    eval_dataset = LemonDataset(eval_file, eval_folder, trans)
    eval_dataloader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True)

    # ネットワークモデル読み込み
    cnn = CNN()
    cnn.load_state_dict(torch.load(model_path))

    # 精度の計算
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, default='../dataset/eval_images.csv')
    parser.add_argument('--eval_folder', type=str, default='../dataset/train_images/')
    parser.add_argument('--model_path', type=str, default='./results/20210305-203632/cifar_net.pth')
    args = parser.parse_args()  # 引数解析
    main()