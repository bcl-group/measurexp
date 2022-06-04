import time
import pandas as pd
import torch
import numpy as np
from tensorly.decomposition import non_negative_parafac_hals
import tensorly as tl
tl.set_backend('pytorch')


def processing(file: str):
    """
    Parameters
    ----------
    file : str
        被験者の pickle ファイル
    """
    print(f'PyTorch バージョン: {torch.__version__}')
    if torch.cuda.is_available():
        print("cuda は有効です。")
    else:
        print("cuda は無効です。")

    df_P = pd.read_pickle(file)

    # セッション番号
    for n_session in np.unique(df_P.reset_index()["Session"]):
        print(f"セッション番号: {n_session:2d}")
        session: pd.DataFrame = df_P.loc[n_session, :, :]
        session.session = n_session

        tensor_data = torch.tensor(
            session.to_numpy(),
            device=torch.device('cuda:0'),
            dtype=torch.double)

        ntf_data: list = [None for _ in range(16)]
        begin_t = time.time()
        for rank in range(16):
            print(f"rank: {rank+1:2d} [{time.time() - begin_t:5.2f}s]")
            ntf_data[rank] = non_negative_parafac_hals(
                tensor_data, rank=rank+1)

        VAF: np.ndarray = np.zeros(16)
        for i in range(16):
            W, H = ntf_data[i][1]
            VAF[i] = 1 - (tensor_data - torch.matmul(W, H.T)
                          ).norm() / tensor_data.norm()
        VAF: pd.DataFrame = pd.DataFrame({"VAF": VAF}, index=np.arange(16) + 1)
        VAF.plot.bar(legend=False, xlim=[1, 16], ylim=[0, 1])

        print(f"保存: session{n_session}-NMF.pickle")
        pd.to_pickle(ntf_data, f"session{n_session}-NMF.pickle")


if __name__ == '__main__':
    pass
