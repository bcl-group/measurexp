# measurexp.muscle_synergy
筋シナジー解析に必要なモジュール

以下の例では、
- h.csv  
  筋シナジーの時間変化データ (サンプル数 x シナジー数)
- w.csv  
  筋シナジーの基底データ (筋肉数 x シナジー数)
- muscles.csv  
  筋肉の名称

> 例 1
```py
from measurexp.muscle_synergy import MuscleSynergy

ms = MuscleSynergy()
ms.read('h.csv', 'w.csv', 'muscles.csv')

# すべての筋シナジーの基底が表示
ms.plot_synergy()
```

> 例 2
```py
from measurexp.muscle_synergy import MuscleSynergy

ms = MuscleSynergy()
ms.read('h.csv', 'w.csv', 'muscles.csv')

# 指定した筋シナジーの基底が表示
ms.plot_synergy(1)
```

