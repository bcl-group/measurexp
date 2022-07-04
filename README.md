# measurexp
<<<<<<< HEAD
[![flake8](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml/badge.svg)](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml)

データ解析ライブラリ

## 対応 Python バージョン
3.9.10
=======
[![flake8](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml/badge.svg?branch=dev)](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml)

データ解析ライブラリ

## 動作環境
- Windows only
- Python 3.9.12
- visual-cpp-build-tools をインストール (https://visualstudio.microsoft.com/visual-cpp-build-tools/)
>>>>>>> dev

Python 用 運動解析ライブラリー

- [筋電筋解析](EMG.md)
- [筋シナジー解析](muscle_synergy.md)

## 規格
- [被験者ファイルについて](about-participants.md)

## 開発環境
```bash
<<<<<<< HEAD
pyenv global 3.9.10
python -m pip install -U pip setuptools
pip install poetry

poetry shell
=======
pyenv global 3.9.12
python -m pip install -U pip setuptools
pip install poetry

python -m poetry shell
python -V # バージョンの確認
>>>>>>> dev
poetry update
```

## ビルド・インストール
```bash
poetry build
poetry install dist/measurexp-0.1.0-py3-none-any.whl
```

### アンインストール
```bash
pip uninstall measurexp -y
```


