# measurexp (Measurement experiments)
[![flake8](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml/badge.svg)](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml)

## 対応 Python バージョン
3.9.10

Python 用 運動解析ライブラリー

- [筋電筋解析](EMG.md)
- [筋シナジー解析](muscle_synergy.md)

## 規格
- [被験者ファイルについて](about-participants.md)

## 開発環境
```bash
pyenv global 3.9.10
python -m pip install -U pip setuptools
pip install poetry

poetry shell
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


