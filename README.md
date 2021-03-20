# stock
==============================

レモンの品質を評価するプログラム

## Requirement
- Python3.7.6

## Usage
### 環境構築
```
cd model
mkdir results
cd ../
pip install -r requirements.txt
PYTHONPATH=/home/stock/
```

### プログラム構成

- train.py: mainとなる学習プログラム
- bi.py: 入出力関係
- preprocess.py: 前処理

train.py の基本的なパラメータは，コマンドライン引数とparam変数で与える．
コマンドライン引数の説明は，
```
python train.py -h
```
で見ることができる．
param で指定するパラメータは，直接プログラム内の変数を書き換える．

出力ファイル群は，resultsディレクトリ内にディレクトリとして出力される．
出力結果は，例えば，以下のようなファイル構成．
```
results/20200608-214943
├── parameter.json  # 使用したパラメータ等
└── cifar_net.pth  # モデルを出力
```

### 学習
- Preprocessing

コマンドライン上から，前処理の種類を指定(グレースケール，平滑化)
```
python preprocessing.py -g -bl
```

- Learning

交差検証を行う
```
python cv.py
```

- 評価

--model_pathでモデルのパスを指定

予測結果として，result.csvを出力
```
python eval.py
```