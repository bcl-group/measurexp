from loguru import logger
import torch
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
        非負値テンソル因子分解を行った結果の配列
        インデックスは rank に対応

    self.emgs: list[EMG.EMG]
        EMG クラスオブジェクトの配列

    self.verbose: bool
        詳細の表示

    Examples
    --------
    >>> csv_dir = '/home/user/experiments/results/202103/A'
    >>> idEMG = IdEMG(verbose=True)
    >>> idEMG.read(csv_dir)
    >>> idEMG.run(rank=3)
    >>> vaf = idEMG.vaf(rank=3)
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Parameter
        ---------
        verbose : bool
            詳細を表示
        """
        self.ranks: list[Any] = [None for _ in range(16)]
        self.ntfs: list[Any] = [None for _ in range(16)]
        self.verbose = verbose

    def get_title(self, filename: str) -> str:
        """ファイル名からタイトルを返します。
        (条件を外部ファイルにまとめるよう修正予定)

        Parameter
        ---------
        filename : str
            ファイル名 (フルパス)

        Return
        ------
        : str
            タイトル
        """
        m = re.search(r"/([A-Z])/EMG-(\d{2})(\d)(\d)\.csv$", filename)
        pid, session, cond_idx, n_condition = m.groups()
        session, cond_idx, n_condition = int(
            session), int(cond_idx), int(n_condition)
        conditions = ['コントロール', '前後能動的揺動', '左右能動的揺動', '前後視覚刺激', '左右視覚刺激']
        return f'被験者 {pid}, ' +\
            f'セッション {session} ' +\
            f'({conditions[cond_idx-1]}条件, ' +\
            f'{n_condition} 回目)'

    @classmethod
    def read_file(cls, args: tuple) -> EMG.EMG:
        file, verbose = args
        if verbose:
            logger.info(f'読み込み中... {file}')
        emg = EMG.EMG()
        emg.read(file)
        emg.set_time(0, 40)
        emg.prep()
        if verbose:
            logger.info(f'読み込み完了... {file}')
        return emg

    def read(self, dir: str) -> 'IdEMG':
        """データ (*.csv) の存在するディレクトリを指定し、
        データを読み込み並列下処理を行います。

        このディレクトリ内のファイルはすべて同一被験者であり、
        階層を持つことはできません。

        Parameter
        ---------
        dir : str
            データが存在するディレクトリ
        """
        file_list: list[str] = glob.glob(f'{dir}/EMG-*.csv')
        with Pool() as pool:
            self.emgs = pool.map(type(self).read_file, [
                                 (_, self.verbose) for _ in file_list])
            if self.verbose:
                logger.info('全データの読み込み完了')
        # pd.DataFrame -> np.ndarray
        self.tensor_data_nd: np.ndarray = np.array([_.rms for _ in self.emgs])
        # np.ndarray -> torch.Tensor
        self.tensor_data: torch.Tensor = torch.tensor(
            self.tensor_data_nd, device='cuda', dtype=torch.float)
        return self

    def vaf(self, rank: int) -> float:
        """VAF を計算します。

        𝐕𝐀𝐅 = |𝑬|² / |𝑿|²

        |𝑿| はフロベニウスノルム。

        Return
        ------
        vaf: float
            VAF の値
        """
        err = tl.norm(self.tensor_data -
                      tl.kruskal_to_tensor(self.ntfs[rank - 1])) ** 2
        return float(1 - err / tl.norm(self.tensor_data) ** 2)

    def run(self, rank: int) -> 'IdEMG':
        """非負値テンソル因子分解 (NTF) を実行します。

        Parameter
        ---------
        rank: int

        Return
        ------
        self: IdEMG
        """
        self.ntfs[rank - 1] = \
            NTF(self.tensor_data, rank=rank, svd='truncated_svd')
        if self.verbose:
            logger.info(f'VAF (rank: {rank}): {self.vaf(rank=rank)}')
        return self


if __name__ == '__main__':
    csv_dir: str = '/mnt/d/experiment/results/202103/A'
    idEMG = IdEMG(verbose=True)
    idEMG.read(csv_dir)
    idEMG.run(rank=3)
