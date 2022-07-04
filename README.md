# measurexp
[![flake8](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml/badge.svg?branch=dev)](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml)

データ解析ライブラリ

## 動作環境
- Windows and Ubuntu (WSL も可)
- Python 3.9 系 (3.9.13)

- [筋電筋解析](EMG.md)
- [筋シナジー解析](muscle_synergy.md)

## 規格
- [被験者ファイルについて](about-participants.md)

## 開発環境

> **Warning**
> 
> Windows (WSL を除く) の場合は、torch の whl ファイルを適切なものに変更する必要があります。

> **Warning**
> 
> Python のバージョン管理ツールである pyenv をインストールしてください。


```bash
python -V # 3.9.13 を確認
python -m pip install -U pip setuptools wheel poetry

cuda_version=$(nvidia-smi | grep "CUDA V" | sed -e s/.*CUDA\ Version:\ //g | sed -e s/[^0-9.]//g)
torch_whl=torch-1.12.0+cu102-cp39-cp39-linux_x86_64.whl
if [ $cuda_version == 10.2 ]; then
    torch_url=https://download.pytorch.org/whl/cu102/torch-1.12.0%2Bcu102-cp39-cp39-linux_x86_64.whl
elif [ $cuda_version == 11.5 ]; then
    torch_whl_=torch-1.11.0+cu115-cp39-cp39-linux_x86_64.whl
    cat pyproject.toml | sed -e s/$torch_whl/$torch_whl_/g > pyproject.toml
    torch_whl=torch_whl_
    torch_url=https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp39-cp39-linux_x86_64.whl
else
    return 1
fi
if [ ! -f $torch_whl ];then wget $torch_url; fi

python -m poetry shell

poetry install
```

> **Note**
> 
> `poetry install` 時にエラーが発生する場合は、`poetry.lock` を削除し、`poetry install` を行ってください。

## リント
```bash
flake8 EMG measurexp
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
