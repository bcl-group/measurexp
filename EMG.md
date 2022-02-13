# measurexp.EMG
筋電位解析に必要なモジュール

以下の例では、
- EMG.csv  
  計測器から出力した筋電位データ

> 例 1
```py
from measurexp.EMG import EMG
import matplotlib.pyplot as plt

emg = EMG()

# データ読み込み
emg.read('EMG.csv')

# 下処理
emg.prep()

# 筋シナジーの計算
emg.calc_synergy()

# 筋シナジーのプロット
emg.plot_synergy()
```
