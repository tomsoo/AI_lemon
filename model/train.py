from bi import read_csv, read_img, write_img
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
        'grayed': True, # グレースケール
        'bright': True, # 輝度調整
        'blur' : True, # 平滑化(フィルター)
        'morph' : False, # 平滑化(モルフォロジー)
        'threshold' : False, # 閾値処理
    })  # 追加パラメータ

    # 実行時のパラメータをファイルとして記録
    with open(f'{result_dir}parameter.json', mode='w') as f:
        json.dump(param, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    # パラメータの読み込み
    train_filename = param['train']
    test_filename = param['test']

    # 画像の読み込み
    train_img, train_label = read_csv(train_filename)

    # 前処理
    img = read_img(train_img, "train", param)
    write_img(img, train_img, timestamp)

    print("Finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='../dataset/train_images.csv')
    parser.add_argument('--test', type=str, default='../dataset/test_images.csv')
    args = parser.parse_args()  # 引数解析
    main()