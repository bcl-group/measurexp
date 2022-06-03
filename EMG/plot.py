from typing import Union
import pandas as pd
import matplotlib.pyplot as plt


def H(df: pd.DataFrame, muscle_colors: list[str], ylim_max: float = 10):
    """筋シナジーの重みを表示します。
    
    Parameters
    ----------
    df : pd.DataFrame
        筋シナジーの重みデータフレーム (n_features, n_components = 筋肉数, 筋シナジー数)
    muscle_colors : list[str]
        各筋肉に対応するカラーコードのリスト
    ylim_max : float = 10
        ylim の最大値
    """
    n_components: int = df.columns.size
    fig, ax = plt.subplots(n_components, figsize=(4.8, 6.4))
    for i in range(n_components):
        df.iloc[:, i].plot.bar(ax=ax[i], ylim=[0, ylim_max], color=muscle_colors)
        if i < n_components - 1:
            ax[i].tick_params(labelbottom=False)
    fig.tight_layout()
    plt.show()


def W(df: pd.DataFrame, loc: Union[int, float, str] = 0, ylim_max: float = 0.3):
    """筋シナジーの時間変化を表示します。

    Parameters
    ----------
    df : pd.DataFrame
        筋シナジーの時間変化のデータフレーム
    loc : int | float | str = 0
        凡例の位置
    ylim_max : float = 0.3
        ylim の最大値
    """
    _, ax = plt.subplots()
    df.plot(ax=ax, xlim=[df.index[0], df.index[-1]], ylim=[0, ylim_max])
    ax.legend(loc=loc)
    plt.show()