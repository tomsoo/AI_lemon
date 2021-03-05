# stock
==============================

レモンの品質を評価するプログラム

## Requirement
- Python3.7.6

## Usage
### 環境構築
```
cd model
mkdir result
cd ../
pip install -r requirements.txt
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

出力ファイル群は，resultディレクトリ内にディレクトリとして出力される．
出力結果は，例えば，以下のようなファイル構成．
```
result/20200608-214943
├── parameter.json  # 使用したパラメータ等
└── preprocess  # 前処理の結果を出力
```

### 学習
- 学習
```
python train.py
```
- 評価
```
python eval.py
```