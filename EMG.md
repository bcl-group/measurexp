# measurexp.EMG
筋電位解析に必要なモジュール

以下の例では、
- EMG.csv  
  計測器から出力した筋電位データ

```mermaid
classDiagram
  class EMG{
    data: pandas.DataFrame
    fs: float
    H: pandas.DataFrame
    W: pandas.DataFrame
    read(filename: str)
    prep(period: float = 0.5, n: int = 5, Wn: int = 50)
    calc_synergy()
    plot_synergy()
  }
```

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
