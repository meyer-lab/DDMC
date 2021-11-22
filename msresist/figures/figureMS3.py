"""
This creates Supplemental Figure 3: Predictive performance of DDMC clusters using different weights
"""

import matplotlib
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((5, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    p = pd.read_csv("msresist/data/Performance/preds_phenotypes_rs_15cl.csv").iloc[:, 1:]
    p.iloc[-3:, 1] = 1250
    xx = pd.melt(p, id_vars=["Run", "Weight"], value_vars=p.columns[2:], value_name="AUC", var_name="Phenotypes")
    sns.lineplot(data=xx, x="Weight", y="AUC", hue="Phenotypes", ax=ax[0])
    ax.legend(prop={'size': 5}, loc=0)

    return f
