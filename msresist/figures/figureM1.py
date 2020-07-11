"""
This creates Figure M1.
"""

from .common import subplotLabel, getSetup
import random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error
from msresist.clustering import MassSpecClustering
from msresist.pre_processing import filter_NaNpeptides

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    d = X.select_dtypes(include=["float64"]).T
    i = X.select_dtypes(include=["object"])

    distance_method = "PAM250"

    #Distribution of missingness per petide
    PlotMissingnessDensity(ax[0], d)

    #Artificial missingness plot
    cd = filter_NaNpeptides(X, cut=1)
    assert True not in np.isnan(cd.iloc[:, 4:]), "There are still NaNs."
    nan_per = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    weights = [0, 0.3, 0.5, 1000]
    ncl = 5

    W = PlotArtificialMissingness(ax[1], cd, weights, nan_per, distance_method, ncl)
    PlotAMwins(ax[2:6], W, weights)

    # Add subplot labels
    subplotLabel(ax)

    return f

def PlotMissingnessDensity(ax, d):
    """Plot amount of missingness per peptide."""
    p_nan_counts = []
    for i in range(d.shape[1]):
        p_nan_counts.append(np.count_nonzero(np.isnan(d[i])))

    sns.distplot(p_nan_counts, 10, ax=ax)
    ax.set_title("Missingness distribution in LUAD")
    ax.set_ylabel("Density")
    ax.set_xlabel("Number of missing observations per peptide")

    # Add Mean
    textstr = "$u$ = " + str(np.round(np.mean(p_nan_counts), 1))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax.text(0.015, 0.95, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)

def PlotArtificialMissingness(ax, x, weights, nan_per, distance_method, ncl):
    """Incorporate different percentages of missing values and compute error between the actual 
    versus cluster average value. Note that this works best with a complete subset of the CPTAC data set. 
    Also note that the wins for all fitted models are returned to be plotting in PlotAMwins"""
    x.index = np.arange(x.shape[0])
    nan_indices = []
    errors = []
    missing, m_ = [], []
    n_weights, w_ = [], []
    winner, winner_ = [], []
    wins = []
    p = ["Sequence", "Data", "Both", "Mix"]
    n = x.iloc[:, 4:].shape[1]
    for per in nan_per:
        md = x.copy()
        m = int(n * per)
        #Generate data set with a given % of missingness
        for i in range(md.shape[0]):
            row_len = np.arange(4, md.shape[1])
            cols = random.sample(list(row_len), m)
            md.iloc[i, cols] = np.nan
            nan_indices.append((i, cols))
        #Fit model, compute error, and store wins
        for j in range(len(weights)):
            n_weights.append(weights[j])
            missing.append(per)
            error, wi = FitModelandComputeError(md, weights[j], x, nan_indices, distance_method, ncl)
            errors.append(error)
            for z in range(len(p)):
                w_.append(weights[j])
                m_.append(per)
                winner_.append(p[z])
                wins.append(wi[z])

    #Plot Error
    X = pd.DataFrame()
    X["Weight"] = n_weights
    X["Missing%"] = missing
    X["Error"] = errors

    #Store Wins
    Y = pd.DataFrame()
    Y["Weight"] = w_
    Y["Missing%"] = m_
    Y["Winner"] = winner_
    Y["Wins"] = wins

    sns.lineplot(x="Missing%", y="Error", data=X, hue="Weight", palette="muted", ax=ax)
    return Y

def FitModelandComputeError(md, weight, x, nan_indices, distance_method, ncl):
    """Fit model and compute error during ArtificialMissingness"""
    i = md.select_dtypes(include=['object'])
    d = md.select_dtypes(include=['float64']).T
    model = MassSpecClustering(i, ncl, SeqWeight=weight, distance_method=distance_method, n_runs=1).fit(d, "NA")
    wins = FindWinIntegers(model.wins_)
    z = x.copy()
    z["Cluster"] = model.labels_
    centers = model.transform(d).T  #Clusters x observations
    errors = []
    for idx in nan_indices:
        v = z.iloc[idx[0], idx[1]]
        c = centers.iloc[z["Cluster"].iloc[idx[0]], np.array(idx[1]) - 4]
        errors.append(mean_squared_error(v, c))
    return np.mean(errors), wins

def PlotAMwins(ax, X, weights):
    """Plot all wins across missingness percentages per weight generated in PlotArtificialMissingness."""
    for i in range(len(ax)):
        d = X[X["Weight"] == weights[i]]
        sns.lineplot(x="Missing%", y="Wins", hue="Winner", data=d, ax=ax[i])
        ax[i].set_title("Weight: " + str(weights[i]))
        ax[i].get_legend().remove()
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2)

def FindWinIntegers(won):
    """Convert wins to integers"""
    seqWin = int(won.split("SeqWins: ")[1].split(" DataWins:")[0])
    dataWin = int(won.split("DataWins: ")[1].split(" BothWin:")[0])
    bothWin = int(won.split("BothWin: ")[1].split(" MixWin:")[0])
    mixWin = int(won.split(" MixWin: ")[1])
    return seqWin, dataWin, bothWin, mixWin