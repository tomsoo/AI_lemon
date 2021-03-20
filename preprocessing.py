import argparse
import json
from bi import get_train_data, get_test_data


#@profile
def main():
    # 実行時パラメータ群
    param = vars(args)  # コマンドライン引数を取り込み

    # パラメータの読み込み
    train_file = param['train_file']
    test_file = param['test_file']

    # 前処理
    print("Preprocessing train image...")
    get_train_data(train_file, param)
    print("Preprocessing test image...")
    get_test_data(test_file, param)

    print("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./dataset/train_images.csv')
    parser.add_argument('--test_file', type=str, default='./dataset/test_images.csv')
    parser.add_argument('-g', '--grayed', action='store_true') # グレースケール
    parser.add_argument('-br', '--bright', action='store_true') # 輝度調整
    parser.add_argument('-bl', '--blur', action='store_true') # 平滑化(フィルター)
    parser.add_argument('-m', '--morph', action='store_true') # 平滑化(モルフォロジー)
    parser.add_argument('-t', '--threshold', action='store_true') # 閾値処理
    args = parser.parse_args()  # 引数解析
    main()