# import numpy as np
import pandas as pd
from scipy import signal
# import matplotlib.pyplot as plt


class ForcePlate:
    def __init__(self):
        self.fs: int
        self.n_samples: int
        self.lpfs: int
        self.df: pd.DataFrame
        self.fdf: pd.DataFrame
        self.pos: list = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        # self.f_pos: list = ['F_Fx', 'F_Fy', 'F_Fz', 'F_Mx', 'F_My', 'F_Mz']
        # self.p_id: str = '210309-01-M'

    def read(self, filename, fs: int = 200) -> 'ForcePlate':
        self.fs = fs
        try:
            df = pd.read_csv(filename, header=None)
        except FileNotFoundError:
            print('ファイルが存在しません。')
            return self
        if df.shape[1] == 7:
            df = df.iloc[:, :-1]
        df.columns = self.pos
        df.loc[:, 'Time [s]'] = df.index / fs
        df.set_index('Time [s]', inplace=True)
        self.df = df
        return self

    def filter(self, fc: int = 12) -> 'ForcePlate':
        fdf = self.df.copy()
        b, a = signal.butter(5, 2 * fc / self.fs, 'low')
        for col in fdf.columns:
            fdf.loc[:, col] = signal.filtfilt(b, a, fdf.loc[:, col])
        self.fdf = fdf
        return self
