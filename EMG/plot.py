# 筋活動や筋シナジーのプロットを行う。

from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
import tensorly
import numpy as np
from PIL import Image
from matplotlib.gridspec import GridSpec


def muscles(df: pd.DataFrame) -> None:
    """
    筋活動を表示します。

    Parameter
    ---------
    df : pd.DataFrame
        セッションのデータ (2 次元)
    """
    fig, ax = plt.subplots(16, 1, figsize=[6.4, 4.8*3])
    fig.suptitle(f"セッション {df.session}")

    tmp_df = df.reset_index()
    tmp_df.drop("Task", axis=1, inplace=True)
    tmp_df.index = pd.Series(tmp_df["index"].to_numpy(
        dtype=float) / 1e9, name="Time [s]")
    tmp_df.drop("index", axis=1, inplace=True)
    for i, column in enumerate(df.columns):
        tmp_df.loc[:, column]\
            .reset_index()\
            .loc[:, ["Time [s]", column]]\
            .set_index("Time [s]")\
            .plot(
                ax=ax[i],
                xlim=[0, 40],
                ylabel=column,
                ylim=[0, 1],
                legend=False)
        ax[i].set_xlabel(None)
        ax[i].tick_params(bottom=False, labelbottom=False)
    fig.tight_layout()
    plt.show()


def muscles_ex(df: pd.DataFrame, muscles: list) -> None:
    """
    筋活動を表示します。

    Parameter
    ---------
    df : pd.DataFrame
        セッションのデータ (2 次元)
    """
    fig, ax = plt.subplots(len(muscles), 1, figsize=[6.4, 4.8*2])
    fig.suptitle(f"セッション {df.session}")

    tmp_df = df.reset_index()
    tmp_df.drop("Task", axis=1, inplace=True)
    tmp_df.index = pd.Series(tmp_df["index"].to_numpy(
        dtype=float) / 1e9, name="Time [s]")
    tmp_df.drop("index", axis=1, inplace=True)

    for i, column in enumerate(muscles):
        tmp_df.reset_index()\
            .loc[:, (["Time [s]"] + column)]\
            .set_index("Time [s]")\
            .plot(ax=ax[i], xlim=[0, 40], ylim=[0, 1], legend=False)
        ax[i].set_xlabel(None)
        ax[i].tick_params(bottom=False, labelbottom=False)
        ax[i].legend(loc=1)
    fig.tight_layout()
    plt.show()


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
    fig, ax = plt.subplots(n_components, sharex=True)
    fig.subplots_adjust(hspace=0)
    if type(ax) != np.ndarray:
        ax: list = [ax]
    for i in range(n_components):
        df.iloc[:, i].plot.bar(
            ax=ax[i], ylim=[0, ylim_max], color=muscle_colors)
        # if i < n_components - 1:
        #     ax[i].tick_params(labelbottom=False)
    # fig.tight_layout()
    plt.show()


def W(df: pd.DataFrame,
      loc: Union[int, float, str] = 0,
      ylim_max: float = 0.3):
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


def to_numpy(t):
    # display(t.type())
    if t.type() == 'torch.cuda.DoubleTensor':
        return t.cpu()
    if t.type() == 'torch.cuda.FloatTensor':
        return t.cpu()
    return t.numpy()


def HW(
        ntf_data: tensorly.cp_tensor.CPTensor,
        muscles: list[str],
        muscle_colors: list[str]):
    """結果を表示

    Parameters
    ----------
    ntf_data : tensorly.cp_tensor.CPTensor
        テンソル分解の結果。
    muscles : list[str]
        筋肉のリスト
    muslces_colors : list[str]
        筋肉に対応する色のリスト
    """
    # 筋シナジーの重みを表示
    synergies: pd.DataFrame = pd.DataFrame(
        to_numpy(ntf_data[1][1]),
        columns=[f"筋シナジー {_+1}" for _ in range(ntf_data[1][1].shape[1])],
        index=muscles
    )
    H(synergies, muscle_colors, 7)

    # 筋シナジーの時間変化を表示
    df = pd.DataFrame(
        to_numpy(ntf_data[1][0]),
        index=pd.to_datetime(np.arange(4000) / 100, unit="s"),
        columns=[f"筋シナジー {_+1}" for _ in range(ntf_data[1][1].shape[1])]
    )
    W(df, ylim_max=0.5)


def HW_with_heatmap(
    participant: str = 'A',
    n_session: int = 4,
    n_synergy: int = 3,
    experiments_dir: str = '',
    muscle_names: list[str] = []
):
    df_P = pd.read_pickle(f"{experiments_dir}/{participant}.pickle")
    session: pd.DataFrame = df_P.loc[n_session, :, :]
    ntf_data = pd.read_pickle(
        f'{experiments_dir}/{participant}-session{n_session:02d}-NMF.pickle'
    )

    X = session.to_numpy().T
    X = X / X.max() * 255
    X = X.astype(np.uint8)

    img = Image.fromarray(X, mode='L')

    fig = plt.figure(dpi=150)
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    gs = GridSpec(5, 4, figure=fig)

    axH = fig.add_subplot(gs[1:, 0])
    axW = fig.add_subplot(gs[0, 1:])

    ax = fig.add_subplot(gs[1:, 1:])
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    img = img.resize(
        (int(bbox.width * fig.dpi), int(bbox.height * fig.dpi)),
        resample=Image.Resampling.NEAREST
    )
    ax.set_xlabel("時刻 [s]")
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, img.size[0], 9))
    ax.set_xticklabels(np.linspace(0, 40, 9, dtype=int))
    ax.imshow(img, cmap='jet')
    # ax.set_yticklabels([''] * 16)

    H = ntf_data[n_synergy - 1][1][1].cpu().numpy()
    H = pd.DataFrame(
        H,
        columns=[f"筋シナジー {_+1}" for _ in range(H.shape[1])],
        index=muscle_names
    )
    H[::-1].plot(kind="barh", ax=axH, legend=False)
    # axH.grid(lw=.5, ls='--')
    axH.set_axisbelow(True)
    axH.set_xticks([])
    axH.set_ylabel("筋肉の種類")

    W = ntf_data[n_synergy - 1][1][0].cpu().numpy()
    axW.plot(W, lw=.8)
    axW.set_yticks([])
    axW.set_xticks([])
    axW.set_xticks(np.arange(0, W.shape[0] + 1, 500))
    axW.set_xticklabels([''] * 9)
    axW.set_xlim([0, W.shape[0]])
    axW.set_ylabel("活動度")
    plt.show()


def muscles_combination(filename: str):
    muscles: list = None
    with open(filename) as f:
        muscles = list(map(lambda x: x.strip().split(" "), f.readlines()))
    return muscles
