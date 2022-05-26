import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import _filters
import cupy as cp
from cupyx.scipy import signal as csignal
import pandas as pd
import re
import glob
from cupyx.scipy import ndimage as cndimage
import plotly.io as pio
pio.renderers.default = "notebook_connected"
# pd.options.plotting.backend = "plotly"
pd.options.plotting.backend = "matplotlib"

plt.rcParams["font.family"] = "IBM Plex Sans JP"


def get_condition_name(filename: str) -> str:
    """
    ファイル名からタスク名を取得します。

    Parameter
    ---------
    filename : str
        ファイル名

    Return
    ------
    : str
        タスク名 (例: セッション: 1, 条件: 1 (1 回目))
    """

    m = re.search("EMG-(\d\d)(\d)(\d)\.csv$", filename)
    n_session, i_condition, n_times = int(m[1]), int(m[2]), int(m[3])
    return f"セッション: {n_session}, 条件: {i_condition} ({n_times} 回目)"

def get_session_number(filename: str) -> int:
    """
    ファイル名からセッション番号を取得します。

    Parameter
    ---------
    filename : str
        ファイル名

    Return
    ------
    : int
        セッション番号
    """
    m = re.search("EMG-(\d\d)(\d)(\d)\.csv$", filename)
    return int(m[1])

def ppt(flist: list[str]) -> pd.DataFrame:
    """
    ファイルリストからデータフレームを読み込みます。
    同一被験者の複数ファイルをリストとして指定します。

    Parameter
    ---------
    flist : list[str]
        ファイルのリスト
        例: ["a.csv", "b.csv", ...]

    Return
    ------
    : pandas.DataFrame
        データフレーム
    """
    p_A = []
    for session in flist:
        df = pd.read_csv(session)
        df["Task"] = get_condition_name(session)
        df["Session"] = get_session_number(session)
        df.set_index(["Session", "Task", "Time [s]"], inplace=True)
        p_A.append(df)
    df_A = pd.concat(p_A)
    return df_A

def get_session(df: pd.DataFrame, session: int, start_time: float = 0, end_time: float = 40) -> pd.DataFrame:
    """
    セッションを取得します

    Parameters
    ----------
    df : pandas.DataFrame
        データ

    session : int
        セッション番号

    start_time : float
        開始時刻

    end_time : float
        終了時刻

    Return
    ------
    pandas.DataFrame
    """

    data = df.loc[(df.index.get_level_values('Time [s]') >= start_time) & (df.index.get_level_values('Time [s]') <= end_time)]
    ret_df = data.loc[session, :, :].copy()
    ret_df.session = session
    return ret_df

def muscle_coef(df, session: int) -> np.ndarray:
    """
    筋活動の係数

    Parameters
    ----------
    df : pandas.DataFrame
        被験者の全セッションのデータ
    
    session : int
        セッション番号

    Returns
    -------
    : numpy.ndarray
        標準偏差の最大値を返す
    """

    k = df.groupby(level=[0, 1]).std().loc[session] / df.groupby(level=[0, 1]).std().max()
    return k.to_numpy()

def _prep(data: pd.Series, sd: float = 0.2) -> pd.Series:
    """
    データの下処理を行います。

    Parameter
    ---------
    data : pd.Series
        筋電位データ (1 次元)

    sd : float = 0.2
        平滑化時の標準偏差

    Return
    ------
    power : pd.Series
        筋活動
    """

    time = data.reset_index().loc[:, "Time [s]"]
    # サンプリング周波数
    fs = (len(time) - 1) / (time[len(time) - 1] - time[0])

    _f, t, Zxx = signal.stft(data, fs=fs)

    x: cp.ndarray = cp.abs(cp.asarray(Zxx))
    for i in range(x.shape[0]):
        mean = cp.mean(x[i, :])
        std = cp.std(x[i, :])
        x[i, :] = (x[i, :] - mean) / std

    # 新しいサンプリング周波数
    fs2 = (t.size - 1) / (t[-1] - t[0])

    # 6 ~= 45Hz, 59 ~= 450Hz
    # 平均値 (1 次元)
    power = cp.mean(x[6:60, :], axis=0)

    # ガウシアンフィルター
    power[:] = cndimage.gaussian_filter1d(power, sd * fs2)

    # 正規化 [0, 1]
    power[:] = (power - power.min()) / (power.max() - power.min())
    power[:] /= power.mean()

    time: pd.Series = pd.Series(t, name="Time [s]")
    power: pd.Series = pd.Series(cp.asnumpy(power), index=time)

    return power

def prep(data: pd.Series, sd: float = 0.2, verbose: bool = False) -> pd.Series:
    """
    データの下処理を行います。

    Parameter
    ---------
    data : pd.Series
        筋電位データ (1 次元)

    sd : float = 0.2
        平滑化時のガウシアンフィルターの標準偏差

    Return
    ------
    power : pd.Series
        筋活動
    """
    time: pd.Series = data.reset_index().loc[:, "Time [s]"]
    # サンプリング周波数
    fs: float = (len(time) - 1) / (time[len(time) - 1] - time[0])

    # フィルター掛け
    b, a = signal.butter(5, np.array([50, 300]) / fs * 2, "band")
    filtered: np.ndarray = signal.filtfilt(b, a, data)
    
    # 平滑化
    power: cp.asarray = cp.asarray(filtered) ** 2
    power[:]          = cndimage.gaussian_filter1d(power, sd * fs)
    power[:]          = cp.sqrt(power)

    # min-max 正規化 [0, 1]
    power[:] = (power - power.min()) / (power.max() - power.min())

    # time: pd.Series = pd.Series(t, name="Time [s]")
    power_se: pd.Series = pd.Series(cp.asnumpy(power), index=time)

    if verbose:
        # fig, ax = plt.subplots(2, 2)
        vdf = pd.DataFrame({
            "生データ": data.to_numpy(),
            "フィルター済みデータ": filtered,
            "筋活動": cp.asnumpy(power)
        }, index=time)
        fig = plt.figure(figsize=(6.4*2, 4.8*2), dpi=150)
        ax1 = plt.subplot2grid((3, 2), (0, 0))
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        ax3 = plt.subplot2grid((3, 2), (1, 0))
        ax4 = plt.subplot2grid((3, 2), (1, 1))
        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        # ax1.plot([1,2,3])

        vdf.plot(y="生データ", ax=ax1, legend=False, xlabel=None)
        vdf.plot(y="フィルター済みデータ", ax=ax3, legend=False)
        vdf.plot(y="筋活動", ax=ax5, legend=False)
        ax2.specgram(vdf.loc[:, "生データ"], Fs=fs)
        ax4.specgram(vdf.loc[:, "フィルター済みデータ"], Fs=fs)

        ax1.set_xlabel("")
        ax1.tick_params(labelbottom=False)
        ax2.set_xlabel("")
        ax2.tick_params(labelbottom=False)
        ax3.set_xlabel("")
        ax4.set_xlabel("")
        fig.tight_layout()
        plt.show()

    return power_se

def prep_old(data: np.ndarray) -> np.ndarray:
    """
    データの下処理を行います。(従来手法)

    Parameter
    ---------
    data : np.ndarray
        筋電位データ (1 次元)

    Return
    ------
    power : nnnp.ndarray
        筋活動
    """

    tmp: cp.ndarray = cp.asarray(data)
    tmp[:] = tmp - tmp.mean()
    tmp[:] = tmp ** 2
    tmp[:] = cndimage.gaussian_filter1d(tmp, 0.2 * 1926)
    tmp[:] = np.sqrt(tmp)
    tmp[:] = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    tmp[:] /= tmp.mean()
    tmp: np.ndarray = cp.asnumpy(tmp)
    return tmp

def plot_muscles(df: pd.DataFrame) -> None:
    """
    筋活動を表示します。

    Parameter
    ---------
    df : pd.DataFrame
        セッションのデータ (2 次元)
    """
    fig, ax = plt.subplots(16, 1, figsize=[6.4, 4.8*3])
    fig.suptitle(f"セッション {df.session}")
    for i, column in enumerate(df.columns):
        df.loc[:, column].plot(ax=ax[i], ylabel=column, xlim=[0, 40], ylim=[0, 5])
        ax[i].set_xlabel(None)
        ax[i].tick_params(bottom=False, labelbottom=False)
    fig.tight_layout()
    plt.show()
