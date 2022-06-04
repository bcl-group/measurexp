import re
# import sys
import os
import pandas as pd
import numpy as np
from scipy import signal
from scipy import ndimage
from sklearn import decomposition
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"


class EMG:
    """
    筋電位を扱うクラス

    Attributes
    ----------
    data : pandas.DataFrame
        筋電位データ

    fs : float
        サンプリング周波数

    H : pandas.DataFrame
        筋シナジーの重み

    W : pandas.DataFrame
        筋シナジーの時間変化

    muscle_colors : str | list = 'tab:blue'
        筋肉に対応する色

    begin_time : float
        対象範囲の開始時間

    end_time : float
        対象範囲の終了時間

    Examples
    --------
    >>> from measurexp.EMG import EMG
    >>> emg = EMG()
    >>> emg.read('EMG.csv')
    >>> emg.set_time(0, 30)
    >>> emg.prep()
    >>> emg.calc_synergy()
    >>> emg.plot_synergy()
    >>> emg.set_colors('../muscle_colors.csv')
    >>> emg.plot_synergy_weights()
    """

    def __init__(self):
        # 筋電位データ
        self.data: pd.DataFrame = None
        # フィルター後のデータ
        self.rms: pd.DataFrame = None
        # 正規化後のデータ
        self.norm: pd.DataFrame = None
        # サンプリング周波数
        self.fs: float = None
        self.H: pd.DataFrame = None
        self.W: pd.DataFrame = None
        self.begin_time: float = None
        self.end_time: float = None
        self.muscles_color: str | list = 'tab:blue'
        self.taskname: str = 'Unnamed'

    def set_colors(self, colors) -> 'EMG':
        """筋肉に対応する色を設定します。

        Parameters
        ----------
        colors : str | list
            筋肉に対応する色のリストまたは文字列

        Returns
        -------
        self : EMG
        """
        if type(colors) in (str, list):
            if os.path.isfile(colors):
                self.muscles_color = pd.read_csv(
                    colors, header=None).values.reshape(-1).tolist()
                return self
            self.muscles_color = colors
        else:
            print('筋肉に対応する色を指定してください。')
        return self

    def _col2muscles_name(self, col_muscle: str):
        match = re.match(r'^(.+):', col_muscle)
        if match is not None:
            muscles_name = match.group(1)
        return muscles_name

    def read(self, filename: str) -> 'EMG':
        """データファイル (*.csv) を読み込みます。

        Parameters
        ----------
        filename : str
            データファイル (*.csv)

        Returns
        -------
        self : EMG
        """
        try:
            # self.data = pd.read_csv(filename,
            # encoding='Shift-JIS', header=116)
            self.data = pd.read_csv(filename)
        except UnicodeDecodeError as err:
            print(err)
            print('デコードできません。正しい文字コードを指定してください。')
        except FileNotFoundError as err:
            print(err)
            print('筋電位データのファイルを指定してください。')

        # インデックスの設定
        self.data.set_index(['Time [s]'], inplace=True)
        self.begin_time, self.end_time = \
            self.data.index[0], self.data.index[-1]
        self.fs = self.data.shape[0] / (self.end_time - self.begin_time)
        return self

    def name(self, taskname: str) -> 'EMG':
        """タスク名を設定します。

        Parameters
        ----------
        taskname : str
            タスク名

        Return
        ------
        self : EMG
        """
        self.taskname = taskname
        return self

    def prep(
        self,
        period: float = 0.2,
        n: int = 5,
        Fc: np.ndarray = np.array([5, 500])
    ) -> 'EMG':
        """データの下処理を行います。

        Parameters
        ----------
        period : float = 0.1
            RMS するウィンドウの範囲 (秒)

        n : int = 5
            ローパスフィルターの次数

        Fc : int = 50
            ローパスフィルターの遮断周波数

        Returns
        -------
        self : EMG
        """
        # b, a = signal.butter(n, Fc / self.fs * 2, 'band', analog=True)
        x = self.data.loc[self.begin_time:self.end_time, :].copy()
        # for m in range(x.shape[1]):
        # x.iloc[:, m] = signal.filtfilt(b, a, x.iloc[:, m])

        x[:] = np.power(x - x.mean(), 2)
        for _ in x:
            x[_][:] = ndimage.gaussian_filter1d(x[_], self.fs * period)
        self.rms = np.sqrt(x)
        return self

    def resample(self, fs: int) -> 'EMG':
        df = self.rms.copy()
        self.rms = pd.DataFrame(
            signal.resample(df.to_numpy(), self.end_time * fs),
            index=(np.arange(self.end_time * fs) / fs),
            columns=df.columns
        )
        df = self.data.copy()
        self.data = pd.DataFrame(
            signal.resample(df.to_numpy(), self.end_time * fs),
            index=(np.arange(self.end_time * fs) / fs),
            columns=df.columns
        )
        self.fs = fs

        return self

    def set_muscles(self, muscles) -> 'EMG':
        self.rms = self.rms.loc[:, muscles]
        self.data = self.data.loc[:, muscles]
        return self

    def _vaf(self, X: np.ndarray, W: np.ndarray, H: np.ndarray):
        VAF = 1 - np.power(X - W.dot(H), 2).sum() / np.power(X, 2).sum()
        return VAF

    def _calc_synergy(self, X: np.ndarray, max_vaf: float):
        VAFs = []
        for n_components in range(1, X.shape[1]):
            model = decomposition.NMF(
                n_components=n_components,
                init='nndsvda',
                random_state=0,
                tol=1e-1
            )
            W = model.fit_transform(X)
            H = model.components_
            VAFs.append(self._vaf(X, W, H))
            if VAFs[-1] > max_vaf:
                break

        return (VAFs, W, H)

    def set_time(self, begin_time: float, end_time: float) -> 'EMG':
        """処理対象時間を設定します。

        Parameters
        ----------
        begin_time : float
            処理範囲の開始時間

        end_time : float
            処理範囲の終了時間

        Returns
        -------
        self : EMG
        """
        self.begin_time, self.end_time = begin_time, end_time
        return self

    def calc_synergy(self, max_vaf: float = 0.9, norm: bool = True) -> 'EMG':
        """筋シナジーを計算します。

        Parameters
        ----------
        max_vaf : float = 0.9
            最大 VAF

        norm : bool = True
            正規化した筋シナジーで計算する

        Returns
        -------
        self : EMG
        """
        if self.norm is None:
            print('正規化されたデータが存在しません。正規化する必要があります。')
            return EMG

        X = self.norm.values if norm else self.rms.values

        self.VAFs, W, H = self._calc_synergy(X, max_vaf)
        self.W = pd.DataFrame(
            W,
            columns=[f'筋シナジー {_+1}' for _ in range(W.shape[1])],
            index=self.rms.index
        )
        self.H = pd.DataFrame(
            H,
            columns=self.rms.columns,
            index=[f'筋シナジー {_+1}' for _ in range(W.shape[1])]
        )
        self.n_synergy = self.H.shape[0]
        return self

    def plot_synergy(self, **kwargs) -> 'EMG':
        """筋シナジーの時間変化を表示します。

        Parameters
        ----------
        kwargs : Any


        Returns
        -------
        self : EMG
        """
        fig, ax = plt.subplots()

        plt_kwargs = {'ax': ax}
        if kwargs is not None:
            plt_kwargs.update(kwargs)
        self.W.plot(**plt_kwargs)
        ax.set_xlim(self.begin_time, self.end_time)
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.show()
        return self

    def plot_synergy_weights(self) -> 'EMG':
        """筋シナジーを表示します。

        Returns
        -------
        self : EMG
        """

        fig, ax = plt.subplots(
            self.n_synergy, figsize=(6.4, 1.6 * self.n_synergy))
        if self.n_synergy == 1:
            ax = [ax]
        for _ in range(self.n_synergy):
            self.H.iloc[_, :].plot.bar(ax=ax[_], color=self.muscles_color)
            if _ == self.n_synergy - 1:
                break
            ax[_].axes.xaxis.set_visible(False)
        fig.tight_layout()
        plt.show()
        return self
