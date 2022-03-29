# >>> from measurexp.NTF import analysis
from measurexp import EMG
import glob
import re
from typing import Any
# Non-negative Tensor Factorization
from tensorly.decomposition import non_negative_parafac_hals as NTF
import numpy as np 
from multiprocessing import Pool
import tensorly as tl
tl.set_backend('pytorch')
import torch
from loguru import logger

# 個人の EMG クラス
class IdEMG:
    """被験者 1 人の筋電位を扱うクラス
    
    Attributes
    ----------
    self.tensor_data_nd: np.ndarray
        筋電位データ (テンソル) **直接変更しないこと**

    self.tensor_data: torch.Tensor
        筋電位データ (テンソル) **直接変更しないこと**

    self.ntfs[list]: list[tensorly.cp_tensor.CPTensor]
        非負値テンソル因子分解を行った結果

    self.emgs: list[EMG.EMG]
        EMG クラスオブジェクトの配列
    """

    def __init__(self) -> None:
        self.ranks: list[Any] = [None for _ in range(16)]
        self.ntfs: list[Any] = [None for _ in range(16)]

    def get_title(self, filename: str) -> str:
        m = re.search('/([A-Z])/EMG-(\d{2})(\d)(\d)\.csv$', filename)
        pid, session, cond_idx, n_condition = m.groups()
        session, cond_idx, n_condition = int(session), int(cond_idx), int(n_condition)
        conditions = ['コントロール', '前後能動的揺動', '左右能動的揺動', '前後視覚刺激', '左右視覚刺激']
        return f'被験者 {pid}, セッション {session} ({conditions[cond_idx-1]}条件, {n_condition} 回目)'

    def read_file(self, args: tuple) -> EMG.EMG:
        file, verbose = args
        if verbose:
            logger.info(f'読み込み中... {file}')
        emg = EMG.EMG()
        emg.read(file)
        emg.set_time(0, 40)
        emg.prep()
        return emg

    def read(self, dir: str, verbose: bool = True) -> 'IdEMG':
        file_list: list[str] = glob.glob(f'{dir}/EMG-*.csv')
        with Pool() as pool:
            self.emgs = pool.map(self.read_file, [(_, verbose) for _ in file_list])
            if verbose:
                logger.info('データの読み込み完了')
        # pd.DataFrame -> np.ndarray
        self.tensor_data_nd: np.ndarray = np.array([_.rms for _ in self.emgs])
        # np.ndarray -> torch.Tensor
        self.tensor_data: torch.Tensor = torch.tensor(self.tensor_data_nd, device='cuda', dtype=torch.float)
        return self

    def vaf(self, rank) -> float:
        err = tl.norm(self.tensor_data - tl.kruskal_to_tensor(self.ntfs[rank - 1])) ** 2
        return float(1 - err / tl.norm(self.tensor_data) ** 2)

    def run(self) -> 'IdEMG':
        rank = 2
        self.ntfs[rank - 1] = \
            NTF(self.tensor_data, rank=rank, svd='truncated_svd')
        return self
    