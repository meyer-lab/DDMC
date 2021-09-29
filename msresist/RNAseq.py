import pandas as pd
from os import listdir

def import_RNAseq():
    names = listdir("msresist/data/RNAseq")
    tpm_table = pd.DataFrame()
    for name in names:
        data = pd.read_csv("msresist/data/RNAseq/" + name, delimiter="\t")
        condition = name[10:-4]
        data = data.set_index("target_id")
        tpm_table = tpm_table.append(data.iloc[:, -1].rename(condition))
    return tpm_table.T.sort_index(axis=1)