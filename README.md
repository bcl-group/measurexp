# measurexp (Measurement experiments)
Python 用 運動解析ライブラリー

- [筋電筋解析](EMG.md)
- [筋シナジー解析](muscle_synergy.md)

## ビルド・インストール
このプロジェクトを Python パッケージとしてインストールできます。

```bash
conda activate
python3 -m pip install --upgrade build twine
python3 -m build
pip install dist/measurexp-0.0.1-py3-none-any.whl
```

### アンインストール
```bash
pip uninstall measurexp -y
```


