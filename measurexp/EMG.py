import re
import sys
import os
import pandas as pd
import numpy as np
from scipy import signal
from sklearn import decomposition
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "IPAexGothic"


class EMG:
    """
    筋電位を扱うクラス
    """
    def __init__(self):
        # 筋電位データ
        self.data: pd.DataFrame = None
        # サンプリング周波数
        self.fs: float = None
        self.H: pd.DataFrame = None
        self.W: pd.DataFrame = None
        self.begin_time: float = None
        self.end_time: float = None
        self.muscles_color: str | list = 'tab:blue'
        self.begin_time_idx: int
        self.end_time_idx: int

    def set_colors(self, colors: str | list) -> 'EMG':
        if type(colors) in (str, list):
            if os.path.isfile(colors):
                self.muscles_color = pd.read_csv(colors, header=None).values.reshape(-1).tolist()
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
        try:
            self.data = pd.read_csv(filename, encoding='Shift-JIS', header=116)
        except UnicodeDecodeError as err:
            print(err)
            print('デコードできません。正しい文字コードを指定してください。')
            sys.exit()
        except FileNotFoundError as err:
            print(err)
            print('筋電位データのファイルを指定してください。')

        col_muscles = self.data.columns.tolist()[1:]
        self.data.columns = ['時間 [s]'] \
            + [self._col2muscles_name(_) for _ in col_muscles]
        self.data.set_index('時間 [s]', inplace=True)
        self.begin_time, self.end_time = self.data.index[[0, -1]]
        self.fs = self.data.shape[0] / (self.end_time - self.begin_time)
        return self

    def _window_rms(self, sig: np.ndarray, fs: float, weight: float):
        emp2 = np.power(sig, 2)
        v = np.ones(int(fs * weight)) / (int(fs) * weight)
        rms = np.sqrt(np.convolve(a=emp2, v=v, mode='same'))
        return rms

    def _col_rms(self, muscle: str, period: float, n: int = 5, Wn: int = 50):
        em = self.data.loc[:, muscle].values.copy()
        em[:] -= em.mean()
        b, a = signal.butter(n, Wn / self.fs * 2, btype='low')
        em[:] = signal.filtfilt(b, a, em)
        end_index = self.data.loc[self.begin_time:self.end_time].values.shape[0]
        em = em[self.begin_time_idx:self.end_time_idx]
        rms = self._window_rms(em, self.fs, period)
        return rms

    def prep(self, period: float = 0.5, n: int = 5, Wn: int = 50) -> 'EMG':
        times = self.data.index
        times = times[self.begin_time_idx:self.end_time_idx]
        # times = times[(times >= self.begin_time) & (times < self.end_time)]
        self.rms = pd.DataFrame(index=times).loc[self.begin_time:self.end_time, :]
        for muscle in self.data.columns[::-1]:
            self.rms.insert(
                0,
                muscle,
                self._col_rms(muscle, period=period, n=n, Wn=Wn)
            )
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
        self.begin_time, self.end_time = begin_time, end_time
        times = self.data.reset_index().loc[:,['時間 [s]']]
        time_idx = times[(times.iloc[:,0] >= self.begin_time) & (times.iloc[:,0] < self.end_time)].index
        self.begin_time_idx, self.end_time_idx = time_idx[0], time_idx[-1]
        return self

    def calc_synergy(self, max_vaf: float = 0.9) -> 'EMG':
        X = self.rms.values
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

    def plot_synergy(self, **args) -> 'EMG':
        fig, ax = plt.subplots()
        self.W.plot(ax=ax, **args)
        ax.set_xlim(self.begin_time, self.end_time)
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.show()
        return self

    def plot_synergy_weights(self) -> 'EMG':
        fig, ax = plt.subplots(self.n_synergy, figsize=(6.4, 1.6 * self.n_synergy))
        if self.n_synergy == 1: ax = [ax]
        for _ in range(self.n_synergy):
            self.H.iloc[_,:].plot.bar(ax=ax[_], color=self.muscles_color)
            if _ == self.n_synergy - 1: break
            ax[_].axes.xaxis.set_visible(False)
        fig.tight_layout()
        plt.show()
        return self
