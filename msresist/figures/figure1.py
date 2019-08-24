"""
This creates Figure 1.
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from ..pre_processing import preprocessing
from .common import subplotLabel, getSetup


path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (2, 3))

    # blank out first axis for cartoon
    # ax[0].axis('off')

    # Read in data
    Y_cv1 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw3.csv')).iloc[:30, :11]
    Y_cv2 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw4.csv')).iloc[:29, :11]

    # Assert that there's no significant influence of the initial seeding density
    Y_cvE3_0 = Y_cv1[Y_cv1["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE3 = Y_cv1[Y_cv1["Elapsed"] == 72].iloc[0, 1:]/Y_cvE3_0
    
    Y_cvE4_0 = Y_cv2[Y_cv2["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE4= Y_cv2[Y_cv2["Elapsed"] == 72].iloc[0, 1:]/Y_cvE4_0

    assert sp.stats.pearsonr(Y_cvE3_0, Y_fcE3)[1] > 0.05
    assert sp.stats.pearsonr(Y_cvE4_0, Y_fcE4)[1] > 0.05

    # Normalize to t=0
    for ii in range(1, Y_cv2.columns.size):
        Y_cv1.iloc[:, ii] /= Y_cv1.iloc[0, ii]
        Y_cv2.iloc[:, ii] /= Y_cv2.iloc[0, ii]

    plotTimeCourse(ax[0:2], Y_cv1, Y_cv2)

    plotEndpoint(ax[2], Y_cv1, Y_cv2)

    plotClustergram(ax[3:5])

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotTimeCourse(axs, Y_cv1, Y_cv2):
    """ Plots the Incucyte timecourse. """
    axs[0].set_title("Experiment 3")
    axs[0].plot(Y_cv1.iloc[:, 0], Y_cv1.iloc[:, 1:])
    axs[0].legend(Y_cv1.columns[1:])
    axs[0].set_ylabel("% Confluency")
    axs[0].set_xlabel("Time (hours)")
    axs[1].set_title("Experiment 4")
    axs[1].plot(Y_cv2.iloc[:, 0], Y_cv2.iloc[:, 1:])
    axs[1].set_ylabel("% Confluency")
    axs[1].set_xlabel("Time (hours)");


def plotEndpoint(ax, Y_cv1, Y_cv2):
    range_ = np.linspace(1, 10, 10)

    Y_fcE3 = Y_cv1[Y_cv1["Elapsed"] == 72].iloc[0, 1:]
    Y_fcE4= Y_cv2[Y_cv2["Elapsed"] == 72].iloc[0, 1:]

    ax.set_title("CV t=72 / t=0")
    ax.set_xticks(np.arange(1,11,1))
    ax.set_xticklabels(Y_cv1.columns[1:])
    ax.bar(range_+0.15, Y_fcE3, width=0.3, align='center', label='Exp3', color = "black")
    ax.bar(range_-0.15, Y_fcE4, width=0.3, align='center', label='Exp4', color = "darkred")
    ax.set_ylabel("% Confluency")


def plotClustergram(axs):
    ABC_mc = preprocessing(motifs=True, FCfilter=True, log2T=True)

    # TODO: Clustermap doesn't show up at the moment, because it wants a whole figure
    g = sns.clustermap(ABC_mc.iloc[:, 2:], method = "single", robust=True)

    p = g.dendrogram_row.reordered_ind
    corr = ABC_mc.iloc[p, 2:].T.corr(method='pearson')
    # Correlation heatmap was really just for exploration. Not including here.
    # axs[1] = sns.heatmap(corr,  vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True, ax=axs[1])
    # axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=80, horizontalalignment='right')
    # axs[1].set_title("Correlation Heatmap");
