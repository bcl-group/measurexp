import pandas as pd
import glob
import re
import tensorly
import numpy as np
import torch
import matplotlib.pyplot as plt


def session_from_NMFfile(filename: str):
    """ファイル名からセッション番号を返す"""
    return int(re.search(r'n(\d+)-NMF.pickle$', filename)[1])


def VAF(
        ntf_data: list[tensorly.cp_tensor.CPTensor],
        tensor_data: torch.Tensor,
        title: str,
        plot: bool = False):
    vaf: np.ndarray = np.zeros(16)
    for i in range(16):
        W, H = ntf_data[i][1]
        vaf[i] = 1 - ((tensor_data - torch.matmul(W, H.T)).norm()
                      ** 2) / (tensor_data.norm() ** 2)
    vaf: pd.DataFrame = pd.DataFrame({"VAF": vaf}, index=np.arange(16) + 1)
    if plot:
        vaf.plot.bar(legend=False, xlim=[1, 16], ylim=[0, 1], title=title)
        plt.plot()
    return vaf


def getVAF(participant: str, dir: str = 'data', plot: bool = False):
    files: list[str] = glob.glob(f"{dir}/{participant}-session*-NMF.pickle")
    df_P = pd.read_pickle(f"{dir}/{participant}.pickle")
    VAFs: list[pd.DataFrame] = [None for _ in files]
    sessions: list[int] = [None for _ in files]
    for i, file in enumerate(files):
        sessions[i] = session_from_NMFfile(file)
        tensor_data: torch.Tensor =\
            torch.tensor(
                df_P.loc[sessions[i], :, :].to_numpy(),
                device=torch.device("cuda:0"),
                dtype=torch.double
            )
        ntf_data: list = pd.read_pickle(file)
        # display(sessions[i], file)
        VAFs[i] = VAF(ntf_data=ntf_data, tensor_data=tensor_data,
                      title=f"被験者 {participant}, セッション: {sessions[i]}")
    # display(sessions)
    VAFs: pd.DataFrame = pd.concat(VAFs, axis=1)
    VAFs.columns = sessions
    VAFs = VAFs.loc[:, np.sort(VAFs.columns)]
    if plot:
        VAFs.T.plot(kind='box')
    return VAFs
