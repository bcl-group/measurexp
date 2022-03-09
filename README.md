# measurexp (Measurement experiments)
[![flake8](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml/badge.svg)](https://github.com/bcl-group/measurexp/actions/workflows/flake8.yml)


Python 用 運動解析ライブラリー

- [筋電筋解析](EMG.md)
- [筋シナジー解析](muscle_synergy.md)

## 規格
- [被験者ファイルについて](about-participants.md)

## ビルド・インストール
### 依存
実行に必要な Python のバージョンは 3.10 以上です。
```bash
conda create --name py310 python=3.10 -y
conda activate py310
```

このプロジェクトを Python パッケージとしてインストールできます。

```bash
conda activate py310
python3 -m pip install --upgrade build twine
python3 -m build
pip install dist/measurexp-0.0.1-py3-none-any.whl
```

### アンインストール
```bash
pip uninstall measurexp -y
```


