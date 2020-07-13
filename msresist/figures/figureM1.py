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
from sklearn.cross_decomposition import PLSRegression
from msresist.figures.figure3 import plotR2YQ2Y, plotPCA
from msresist.clustering import MassSpecClustering
from msresist.pre_processing import filter_NaNpeptides

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12.5, 12), (4, 3))
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

    #Wins across different weights with 0.5% missingness
    X_w = filter_NaNpeptides(X, cut=0.5)
    d_w = X_w.select_dtypes(include=['float64']).T
    i_w = X_w.select_dtypes(include=['object'])
    weights = np.arange(0, 1.1, 0.1)
    PlotWinsByWeight(ax[6], i_w, d_w, weights, distance_method, ncl)

    #Leave blank for heatmap of cluster centers across patients
    ax[8].axis('off')

    #Run model
    X_f = filter_NaNpeptides(X, cut=0.1)
    d_f = X_f.select_dtypes(include=['float64']).T
    i_f = X_f.select_dtypes(include=['object'])
    distance_method = "PAM250"
    ncl = 15
    SeqWeight = 0.20
    MSC = MassSpecClustering(i_f, ncl, SeqWeight=SeqWeight, distance_method=distance_method, n_runs=1).fit(d_f, "NA")
    centers = MSC.transform(d_f)
    centers["Patient_ID"] = X.columns[4:]

    #PCA of model
    centers.iloc[:, :-1] = zscore(centers.iloc[:, :-1], axis=1)
    centers = TumorType(centers)
    c = 2
    plotPCA(
        ax[7:11], centers, c, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", hue_load="Cluster"
    )

    #Regress against survival
    centers, y = TransformCPTACdataForRegression(MSC, d_f, list(X.columns[4:]))

    centers_T = centers[~centers["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")
    centers_N = centers[centers["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")
    y_T = y[~y["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")
    y_N = y[y["Patient_ID"].str.endswith(".N")].set_index("Patient_ID")

    plsr = PLSRegression(n_components=2, scale=True)
    plotR2YQ2Y(ax[11], plsr, centers_T, y_T, 1, 10)

    # Add subplot labels
    subplotLabel(ax)

    return f

def FormatWhiteNames(X):
    """Keep only the gene name."""
    genes = []
    counter = 0
    for v in X.iloc[:, 0]:
        if "GN" not in v:
            counter += 1
            continue
        genes.append(v.split("GN=")[1].split(" PE")[0].strip())
    print("number of proteins without gene name:", counter)
    return genes

def FindMatchingPeptides(X, Y, cols=False):
    """Return peptides within the entire CPTAC LUAD data set also present in the White data or vice versa. Note
    that if the white lab data set is used as X, the patient labels should be passed."""
    if cols:
        X = X[cols]
    X = X.dropna().sort_values(by="Sequence")
    X = X.set_index(["Gene", "Sequence"])
    rows = []
    counter = 0
    for idx in range(Y.shape[0]):
        try:
            r = X.loc[Y["Gene"][idx], Y["Sequence"][idx]].reset_index()
            if len(r) > 1:
                rows.append(pd.DataFrame(r.iloc[0, :]).T)
            else:
                rows.append(r)
        except:
            counter += 1
            continue
    print("Number of mismatches: ", counter)

    y = pd.concat(rows)
    return y.drop_duplicates(list(y.columns), keep="first")

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

def PlotWinsByWeight(ax, i, d, weigths, distance_method, ncl):
    """Plot sequence, data, both, or mix score wins when fitting across a given set of weigths. """
    wins = []
    prioritize = []
    W = []
    for w in weigths:
        model = MassSpecClustering(i, ncl, SeqWeight=w, distance_method=distance_method, n_runs=1).fit(d, "NA")
        won = model.wins_
        wi = FindWinIntegers(won)
        W.append(w)
        wins.append(wi[0])
        prioritize.append("Sequence")
        W.append(w)
        wins.append(wi[1])
        prioritize.append("Data")
        W.append(w)
        wins.append(wi[2])
        prioritize.append("Both")
        W.append(w)
        wins.append(wi[3])
        prioritize.append("Mix")

    X = pd.DataFrame()
    X["Sequence_Weighting"] = W
    X["Prioritize"] = prioritize
    X["Wins"] = wins
    sns.lineplot(x="Sequence_Weighting", y="Wins", data=X, hue="Prioritize", ax=ax)

def FindWinIntegers(won):
    """Convert wins to integers"""
    seqWin = int(won.split("SeqWins: ")[1].split(" DataWins:")[0])
    dataWin = int(won.split("DataWins: ")[1].split(" BothWin:")[0])
    bothWin = int(won.split("BothWin: ")[1].split(" MixWin:")[0])
    mixWin = int(won.split(" MixWin: ")[1])
    return seqWin, dataWin, bothWin, mixWin

def TumorType(centers):
    """Add Normal vs Tumor column."""
    tumortype = []
    for i in range(centers.shape[0]):
        if ".N" in centers["Patient_ID"][i]:
            tumortype.append("Normal")
        else:
            tumortype.append("Tumor")
    centers["Type"] = tumortype
    return centers

def TransformCPTACdataForRegression(model, d, patient_IDs):
    """Match patient IDs to clinical outcomes for regression and return phosphoproteomic and clinical data sets."""
    centers = model.transform(d)
    centers["Patient_ID"] = patient_IDs

    #Import Vital Status after 12 months and transform to binary
    cf = pd.read_csv("msresist/data/MS/CPTAC/CPTACLUAD_VitalStatus.csv")
    cf = cf.replace("Living", 0)
    cf = cf.replace("Deceased", 1)

    #Import dictionary with sample IDs to map patients in both data sets
    IDict = pd.read_csv("msresist/data/MS/CPTAC/IDs.csv", header=0)
    IDict_ = dict(zip(IDict.iloc[:, 0], IDict.iloc[:, 1]))
    cf = SwapPatientIDs(cf, IDict_)
    cf = AddTumorPerPatient(cf)

    centers = centers.set_index("Patient_ID").sort_values(by="Patient_ID")
    y = cf.sort_values(by="Patient_ID").set_index("Patient_ID")
    
    #Check differences in patients present in both data sets
    diff = list(set(centers.index) - set(y.index))
    centers = centers.drop(diff)
    assert len(diff) < 10, "missing many samples"

    #Drop patients for which there is no vital status and assert all patient IDs match
    nans = y[np.isnan(y["vital_status_12months"])].index
    y = y.dropna().reset_index()
    centers = centers.drop(nans).reset_index()
    assert all(centers.index == y.index), "samples not matching."
    return centers, y

def SwapPatientIDs(cf, IDict_):
    """Change patient IDs from Case ID to Broad ID."""
    ids = []
    for i in range(cf.shape[0]):
        ids.append(IDict_[cf.iloc[i, 0]])
    cf["Patient_ID"] = ids
    return cf

def AddTumorPerPatient(cf):
    """Add Tumor row per patient in vital status data."""
    for i in range(cf.shape[0]):
        id_ = cf.iloc[i, 0]
        if ".N" in id_:
            iD = id_.split(".N")[0]
            cf.loc[-1] = [iD, cf.iloc[i, 1]]
            cf.index = cf.index + 1
        else:
            continue
    cf.index = cf.index + 1
    return cf.sort_index()