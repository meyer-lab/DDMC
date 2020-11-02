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
            sns.lineplot(x="Elapsed", y="Fold-change confluency", hue="Inh_concentration", data=m, ax=ax[i][j])
            ax[i][j].set_title(line + "-" + treatment)
            if i != 0 or j != 0:
                ax[i][j].get_legend().remove()


def transform_YAPviability_data(X):
    X = pd.melt(X, id_vars="Elapsed", value_vars=X.columns[1:], var_name="Lines", value_name="Fold-change confluency")
    X["Condition"] = [s.split(" ")[1].split(" ")[0] for s in X["Lines"]]
    X["Inh_concentration"] = [s[4:].split(" ")[1] for s in X["Lines"]]
    X["Lines"] = [s.split(" ")[0] for s in X["Lines"]]
    X = X[["Elapsed", "Lines", "Condition", "Inh_concentration", "Fold-change confluency"]]
    return X
