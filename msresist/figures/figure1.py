"""
This creates Figure 1.
"""
import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from ..pre_processing import preprocessing, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd, MergeDfbyMean
from .common import subplotLabel, getSetup


path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (3, 3))

    # blank out first axis for cartoon
    # ax[0].axis('off')

    # Read in data
    Y_cv1 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw3.csv')).iloc[:30, :11]
    Y_cv2 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw4.csv')).iloc[:29, :11]

    # Assert that there's no significant influence of the initial seeding density
    Y_cvE3_0 = Y_cv1[Y_cv1["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE3 = Y_cv1[Y_cv1["Elapsed"] == 72].iloc[0, 1:]/Y_cvE3_0
    
    Y_cvE4_0 = Y_cv2[Y_cv2["Elapsed"] == 0].iloc[0, 1:]
    Y_fcE4 = Y_cv2[Y_cv2["Elapsed"] == 72].iloc[0, 1:]/Y_cvE4_0

    assert sp.stats.pearsonr(Y_cvE3_0, Y_fcE3)[1] > 0.05
    assert sp.stats.pearsonr(Y_cvE4_0, Y_fcE4)[1] > 0.05

    # Normalize to t=0
    for ii in range(1, Y_cv2.columns.size):
        Y_cv1.iloc[:, ii] /= Y_cv1.iloc[0, ii]
        Y_cv2.iloc[:, ii] /= Y_cv2.iloc[0, ii]

    plotTimeCourse(ax[0:2], Y_cv1, Y_cv2)

    plotAveragedEndpoint(ax[2], Y_cv1, Y_cv2)

    plotRTKs(ax[3:7])

    plotVarReplicates(ax[7:9])

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotTimeCourse(ax, Y_cv1, Y_cv2):
    """ Plots the Incucyte timecourse. """
    ax[0].set_title("Experiment 3")
    ax[0].plot(Y_cv1.iloc[:, 0], Y_cv1.iloc[:, 1:])
    ax[0].legend(Y_cv1.columns[1:])
    ax[0].set_ylabel("Fold-change to t=0h")
    ax[0].set_xlabel("Time (hours)")
    ax[1].set_title("Experiment 4")
    ax[1].plot(Y_cv2.iloc[:, 0], Y_cv2.iloc[:, 1:])
    ax[1].set_ylabel("Fold-change to t=0h")
    ax[1].set_xlabel("Time (hours)");


def plotReplicatesEndpoint(ax, Y_cv1, Y_cv2):
    range_ = np.linspace(1, 10, 10)

    Y_fcE3 = Y_cv1[Y_cv1["Elapsed"] == 72].iloc[0, 1:]
    Y_fcE4 = Y_cv2[Y_cv2["Elapsed"] == 72].iloc[0, 1:]

    ax.set_title("Cell Viability")
    ax.set_xticks(np.arange(1,11,1))
    ax.set_xticklabels(Y_cv1.columns[1:])
    ax.bar(range_+0.15, Y_fcE3, width=0.3, align='center', label='Exp3', color = "black")
    ax.bar(range_-0.15, Y_fcE4, width=0.3, align='center', label='Exp4', color = "darkred")
    ax.set_ylabel("Fold-change 72h vs 0h")


def plotAveragedEndpoint(ax, Y_cv1, Y_cv2):
    range_ = np.linspace(1, 10, 10)

    Y_cv = MergeDfbyMean(pd.concat([Y_cv1, Y_cv2], axis=0), Y_cv1.columns, "Elapsed")
    Y_cv = Y_cv.reset_index()[Y_cv1.columns]
    Y_cv = Y_cv[Y_cv["Elapsed"] == 72].iloc[0, 1:]

    ax.set_title("Cell Viability")
    ax.set_xticks(np.arange(1,11,1))
    ax.set_xticklabels(Y_cv1.columns[1:])
    ax.bar(range_, Y_cv, width=0.5, align='center', label='Exp3', color = "black")
    ax.set_ylabel("Fold-change 72h vs 0h")


def plotRTKs(ax):
    ABC = preprocessing()
    header = ABC.columns

    EGFR = ABC[ABC["Sequence"].str.contains("SHQISLDNPDyQQDFFP")].mean()
    IGFR = ABC[ABC["Sequence"].str.contains("IYETDYyR")].iloc[:, 3:13].mean()
    MET = ABC[ABC["Sequence"].str.contains("MYDkEyYSVHNk")].iloc[:, 3:13].mean()
    AXL = ABC[ABC["Sequence"].str.contains("YNGDyYR")].iloc[:, 3:13].mean()

    ax[0].set_title("EGFR: pY1172", fontsize=13)
    ax[0].plot(EGFR)
    ax[0].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')
    ax[1].set_title("IGFR: pY1190", fontsize=13)
    ax[1].plot(IGFR)
    ax[1].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')
    ax[2].set_title("MET: pY1234", fontsize=13)
    ax[2].plot(MET)
    ax[2].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')
    ax[3].set_title("AXL: pY702", fontsize=13)
    ax[3].plot(AXL)
    ax[3].set_xticklabels(header[3:], rotation=80, horizontalalignment='right')


def plotVarReplicates(ax):
    ABC = preprocessing(rawdata=True)
    dups = pd.pivot_table(ABC, index=['Protein', 'Sequence'], aggfunc="size").sort_values()
    NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

    #Correlation of Duplicates, optionally filtering first
    DupsTable = BuildMatrix(CorrCoefPeptides, ABC)
    # DupsTable = CorrCoefFilter(DupsTable)
    DupsTable_drop = DupsTable.drop_duplicates(["Protein", "Sequence"])
    assert(DupsTable.shape[0]/2 == DupsTable_drop.shape[0])

    #Stdev of Triplicates, optionally filtering first
    StdPeptides = BuildMatrix(StdPeptides, ABC)
    TripsTable = TripsMeanAndStd(StdPeptides, list(ABC.columns[:4]))
    Stds = TripsTable.iloc[:, TripsTable.columns.get_level_values(1) == 'std']
    # Xidx = np.all(Stds.values <= 0.4, axis=1)
    # Stds = Stds.iloc[Xidx, :]
    Std_mean = Stds.iloc[:, 1:].mean(axis=1)

    n_bins = 10
    ax[0].hist(DupsTable_drop.iloc[:, 12], bins=n_bins)
    ax[0].set_ylabel("Number of peptides", fontsize=12)
    ax[0].set_xlabel("Pearson Correlation Coefficients", fontsize=12)
    ax[1].hist(Std_mean, bins=n_bins)
    ax[1].set_xlabel("Mean of Standard Deviations", fontsize=12)
    ax[1].set_ylabel("Number of peptides", fontsize=12)


# TODO: Clustermap doesn't show up at the moment, because it wants a whole figure    
# def plotClustergram(ax):
#     ABC_mc = preprocessing(motifs=True, FCfilter=True, log2T=True)

#     g = sns.clustermap(ABC_mc.iloc[:, 2:], method = "single", robust=True)

#     p = g.dendrogram_row.reordered_ind
#     corr = ABC_mc.iloc[p, 2:].T.corr(method='pearson')
    # Correlation heatmap was really just for exploration. Not including here.
    # ax[1] = sns.heatmap(corr,  vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True, ax=ax[1])
    # ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=80, horizontalalignment='right')
    # ax[1].set_title("Correlation Heatmap");
