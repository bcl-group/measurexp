---
sidebar_position: 1
---

# はじめに
このリポジトリは運動解析のための Python ライブラリです。
使用はもちろん利用 (改変等) も可能です。

### 開発及びインストールに必要な環境
このリポジトリは Python 3.9.10 での使用を想定していますが、他のバージョンでもできるかもしれません。

```sh
pyenv global 3.9.10
```

Python のパッケージ管理ツールである Poetry に依存しているため、
開発およびビルド行う場合は Poetry のインストールが必要です。

```sh
$ python -m pip install -U pip setuptools

# poetry のインストール
$ python -m pip install poetry

# 仮想環境内のシェルに入る
$ poetry shell

# 依存するパッケージのインストール
(.venv) $ poetry update
```

## ビルドおよびインストール
パッケージのビルドは次のように行います。ビルドを行うことにより、pip や poetry のパッケージとして利用が可能です。
```sh
(.venv) $ poetry build
```

ビルドしたパッケージのインストールは次のように行います。
```sh
poetry add dist/measurexp-0.1.0.tar.gz
```

pip の場合は次のようにします。

```sh
pip install dist/measurexp-0.1.0.tar.gz
```

## アンインストール
パッケージのアンインストールは通常の方法で行います。

```sh
pip uninstall measurexp -y
```
