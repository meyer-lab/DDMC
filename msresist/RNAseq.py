import pandas as pd
from os import listdir

def import_RNAseq():
    names = listdir("msresist/data/RNAseq") #data currently not stored
    tpm_table = pd.DataFrame()
    for name in names:
        data = pd.read_csv("msresist/data/RNAseq/" + name, delimiter="\t")
        condition = name[10:-4]
        data = data.set_index("target_id")
        tpm_table = tpm_table.append(data.iloc[:, -1].rename(condition))
    tpm_table = tpm_table.T.sort_index(axis=1).T
    tpm_table["Sum"] = tpm_table.sum(axis=1)
    tpm_table = tpm_table[tpm_table["Sum"] > 0].drop("Sum", axis=1)
    tpm_table = tpm_table.reset_index()
    tpm_table = tpm_table.rename(columns={"index": "Cell Line"}).reset_index()
    tpm_table.to_feather("msresist/data/AXLmutants_RNAseq_feathered.csv")