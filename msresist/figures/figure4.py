"""
This creates Figure 4.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from msresist.figures.figure1 import TimePointFoldChange


def plot_YAPinhibitorTimeLapse(ax, X):
    lines = ["WT", "KO"]
    treatments = ["UT", "E", "E/R", "E/A"]
    for i, line in enumerate(lines):
        for j, treatment in enumerate(treatments):
            m = X[X["Lines"] == line]
            m = m[m["Condition"] == treatment]
            sns.lineplot(x="Elapsed", y="Fold-change confluency", hue="Inh_concentration", data=m, ci=68, ax=ax[i][j])
            ax[i][j].set_title(line + "-" + treatment)
            if i != 0 or j != 0:
                ax[i][j].get_legend().remove()


def transform_YAPviability_data(data, itp=12):
    """Transform to initial time point and convert into seaborn format"""
    new = []
    for i, mat in enumerate(data):
        if i > 0:
            mat = MeanTRs(mat)
        new.append(TimePointFoldChange(mat, itp))

    c = pd.concat(new, axis=0)
    c = pd.melt(c, id_vars="Elapsed", value_vars=c.columns[1:], var_name="Lines", value_name="Fold-change confluency")
    c["Condition"] = [s.split(" ")[1].split(" ")[0] for s in c["Lines"]]
    c["Inh_concentration"] = [s[4:].split(" ")[1] for s in c["Lines"]]
    c["Lines"] = [s.split(" ")[0] for s in c["Lines"]]
    c = c[["Elapsed", "Lines", "Condition", "Inh_concentration", "Fold-change confluency"]]
    c = c[c["Elapsed"] >= itp]
    return c


def MeanTRs(X):
    """Merge technical replicates of 2 BR by taking the mean."""
    idx = [np.arange(0, 6) + i for i in range(1, X.shape[1], 12)]
    for i in idx:
        for j in i:
            X.iloc[:, j] = X.iloc[:, [j, j + 6]].mean(axis=1)
            X.drop(X.columns[j + 6], axis="columns")

    return X.drop(X.columns[[j + 6 for i in idx for j in i]], axis="columns")
