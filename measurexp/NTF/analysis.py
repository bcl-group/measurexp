from loguru import logger
import torch
from measurexp import EMG
import glob
import re
import hashlib
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
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
    >>> from measurexp.NTF.analysis import IdEMG
    >>> csv_dir = '/home/user/experiments/results/202103/A'
    >>> idEMG.read(csv_dir, verbose=False)
    >>> idEMG.runall()
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Parameter
        ---------
        verbose : bool
            詳細を表示
        """
        self.vafs: list[Any] = [None for _ in range(16)]
        self.ntfs: list[Any] = [None for _ in range(16)]
        self.verbose = verbose
        self.color = 'tab:blue'

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

    def files_stat_hash(self, file_list: list[str]) -> str:
        """ファイルのメタデータを MD5 でハッシュ化します。
        """
        stat_txt: str = ''
        for file_name in file_list:
            p = pathlib.Path(file_name)
            stat_txt += str(p.stat())
        hash = hashlib.md5(stat_txt.encode())
        return hash.hexdigest()

    @classmethod
    def read(
        cls,
        dir: str,
        verbose: bool = False,
        cache: bool = True
    ) -> 'IdEMG':
        """データ (EMG-*.csv) の存在するディレクトリを指定し、
        データを読み込み並列下処理を行います。
        キャッシュが有効化されている場合、キャッシュから読み込みます。

        このディレクトリ内のファイルはすべて同一被験者であり、
        階層を持つことはできません。

        Parameter
        ---------
        dir : str
            データが存在するディレクトリ

        verbose : bool
            詳細を表示

        cache : bool
            キャッシュの有効化

        : IdEMG
            IdEMG のインスタンス
        """
        self = cls()
        self.verbose = verbose
        file_list: list[str] = glob.glob(f'{dir}/EMG-*.csv')
        if len(file_list) == 0:
            logger.error('ファイルが存在しません。')
            raise FileNotFoundError
        data_cache = f'/tmp/{self.files_stat_hash(file_list)}.pkl'
        if os.path.isfile(data_cache) and cache:
            logger.info('キャッシュから読み込んでいます。')
            self = pd.read_pickle(data_cache)
        else:
            if not cache:
                logger.info('キャッシュが無効化されています')
            with Pool() as pool:
                self.emgs = pool.map(type(self).read_file, [
                                    (_, self.verbose) for _ in file_list])
                if self.verbose:
                    logger.info('全データの読み込み完了しました。')
            # pd.DataFrame -> np.ndarray
            self.tensor_data_nd: np.ndarray = np.array(
                [_.rms for _ in self.emgs])
            # np.ndarray -> torch.Tensor
            self.tensor_data: torch.Tensor = torch.tensor(
                self.tensor_data_nd, device='cuda', dtype=torch.float)
            pd.to_pickle(self, data_cache)
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
        rank : int

        Return
        ------
        self : IdEMG
        """
        self.ntfs[rank - 1] = \
            NTF(self.tensor_data, rank=rank, svd='truncated_svd')
        self.vafs[rank - 1] = self.vaf(rank=rank)
        if self.verbose:
            logger.info(f'VAF (rank: {rank}): {self.vafs[rank - 1]}')
        return self

    def runall(self, t_VAF: float = 0.9) -> 'IdEMG':
        """VAF の閾値まで非負値テンソル因子分解を実行します。

        Parameter
        ---------
        t_VAF : float
            VAF の閾値

        Return
        ------
        self : IdEMG
        """
        self.t_VAF = t_VAF
        for r in range(16):
            rank = r + 1
            self.run(rank=rank)
            if self.vafs[r] > t_VAF:
                break
        return self

    def plot_vaf(self) -> 'IdEMG':
        """VAF をプロットします。
        runall メソッドを呼び出し後に実行する必要があります。

        Return
        ------
        self : IdEMG | None
        """
        vaf = np.array(list(filter(None, self.vafs)))
        if len(vaf) == 0:
            logger.error('NTF が行われていません。`runall` を実行する必要があります。')
            return None
        _, ax = plt.subplots()
        ax.hlines(0.9, 1, len(vaf), color='tab:gray',
                  linestyles='--', linewidth=0.5)
        ax.plot([_ + 1 for _ in range(len(vaf))], vaf)
        ax.scatter([_ + 1 for _ in range(len(vaf))], vaf)
        ax.set_ylim([0, 1])
        ax.set_xlim([1, len(vaf)])
        ax.set_xticks([_ + 1 for _ in range(len(vaf))])
        ax.set_ylabel('Variance Accounted For (VAF)')
        ax.set_xlabel('Number of muscle synergies')
        plt.show()
        return IdEMG

    def plot_synergy(self, rank: int, **kwargs):
        df = pd.DataFrame(self.ntfs[rank-1][1][2].cpu())
        df.index = self.muscles_list
        for s in range(len(df.columns)):
            df[s].plot.bar(figsize=(6.4, 1.2), color=self.color, **kwargs)
            plt.show()
        return self


if __name__ == '__main__':
    csv_dir: str = '/mnt/d/experiment/results/202103/A'
    idEMG = IdEMG(verbose=True)
    idEMG.read(csv_dir)
    idEMG.runall()
