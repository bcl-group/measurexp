import pandas as pd
import numpy as np
import cupy as cp
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"

file = "C:\\Users\\hikari\\EMG-0441.csv"

df = pd.read_csv(file, index_col="Time [s]")
x = df["ヒラメ筋"].to_numpy()

def get_fs(x: pd.DataFrame):
    n = len(x.index)
    end_t = x.index[-1]
    # サンプリング周波数
    fs = (n - 1) / end_t
    return fs

fs = get_fs(df)

zxx, f, t, _ = plt.specgram(x, Fs=fs)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(f[:, None], t[None, :], 10.0*np.log10(zxx), cmap=cm.coolwarm)
plt.show()

