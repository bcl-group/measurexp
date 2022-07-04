# EMG 下処理の試験コード
# python -m measurexp.non-parallel-emg

from measurexp import EMG
import importlib
importlib.reload(EMG)
<<<<<<< HEAD
import pandas as pd
=======
# import pandas as pd
>>>>>>> dev
# from multiprocessing import Pool
# import glob

if __name__ == '__main__':
    file: str = "/mnt/d/experiment/results/202103/A/EMG-0111.csv"
    emg = EMG.EMG()
    emg.verbose = True
    emg.read(file)
    emg.set_time(0, 40)
    emg.prep()
<<<<<<< HEAD

=======
>>>>>>> dev
