import pandas as pd
from multiprocessing import Pool
import glob

from measurexp import EMG
import importlib
importlib.reload(EMG)


def prepMultiEMG(file: str):
    return EMG.EMG() \
        .read(file) \
        .set_time(0, 40) \
        .prep()


def getEMGs(dir, muscles):
    EMGs: list[EMG.EMG]

    with Pool(8) as p:
        files = glob.glob(f"{dir}/EMG-*.csv")
        EMGs = p.map(prepMultiEMG, files)

    for i in range(8):
        tmp1, tmp2 = EMGs[i].data, EMGs[i].rms
        EMGs[i].data = tmp1.loc[:, muscles]
        EMGs[i].rms = tmp2.loc[:, muscles]
        del tmp1, tmp2

    EMG_maxs = list(range(len(EMGs)))
    EMG_mins = list(range(len(EMGs)))
    for i, _ in enumerate(EMGs):
        EMG_maxs[i] = _.rms.max()
        EMG_mins[i] = _.rms.min()
    EMG_max = pd.DataFrame(EMG_maxs).max()
    EMG_min = pd.DataFrame(EMG_mins).min()
    for i, _ in enumerate(EMGs):
        EMGs[i].norm = ((EMGs[i].rms - EMG_min) / (EMG_max - EMG_min))
    return EMGs


def get_EMGs(participants_dir: str,
             muscles_file: str = "muscles_list.csv"
             ) -> list[EMG.EMG]:
    muscles: list[str] = pd.read_csv(muscles_file, header=None).to_numpy()[
        :, -1].tolist()
    EMGs: list[EMG.EMG] = getEMGs(participants_dir, muscles)
    return EMGs
