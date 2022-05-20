import pandas as pd
import matplotlib.pyplot as plt
from measurexp.NTF import analysis
import numpy as np
plt.rcParams["font.family"] = "IPAexGothic"

# サンプリング周波数を求める
def get_fs(x: pd.DataFrame):
    n = len(x.index)
    end_t = x.index[-1]
    # サンプリング周波数
    fs = (n - 1) / end_t
    return fs

csv_dir: str = '/mnt/d/experiment/results/202103/A'
conditions_list = ["コントロール", "前後", "左右", "前後視覚刺激", "左右視覚刺激"]
muscles = pd.read_csv('muscles_list.csv', header=None).values[:,-1].tolist()
idEMG = analysis.IdEMG.read(csv_dir, muscles, conditions_list, verbose=True, cache=True)
x = idEMG.emgs[1].data["ヒラメ筋"].to_numpy()



x = idEMG.emgs[1].data["ヒラメ筋"][0:40]
fs = get_fs(x)

zxx, f, t, _ = plt.specgram(x, Fs=fs)

from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(f[:, None], t[None, :], 10.0*np.log10(zxx), cmap=cm.coolwarm)
plt.show()

