"""
This creates Figure 4: Predictive performance of DDMC clusters using different weights
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_squared_error
from ..clustering import DDMC
from .common import subplotLabel, getSetup, find_patients_with_NATandTumor, TransformCenters, merge_binary_vectors, HotColdBehavior
from ..logistic_regression import plotROC
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 2), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=0.5, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=2)
    d = X.select_dtypes(include=[float]).T
    i = X.select_dtypes(include=[object])

    # Plot mean AUCs per model
    p = pd.read_csv("msresist/data/Performance/preds_phenotypes_rs_15cl.csv").iloc[:, 1:]
    p.iloc[-3:, 1] = 1250
    xx = pd.melt(p, id_vars=["Run", "Weight"], value_vars=p.columns[2:], value_name="AUC", var_name="Phenotypes")
    sns.lineplot(data=xx, x="Weight", y="AUC", hue="Phenotypes", ax=ax[0])
    ax[0].legend(prop={'size': 5}, loc=0)

    # Fit Data, Mix, and Seq Models
    dataM = DDMC(i, ncl=30, SeqWeight=0, distance_method="Binomial", random_state=5).fit(d)
    mixM = DDMC(i, ncl=30, SeqWeight=250, distance_method="Binomial", random_state=5).fit(d)
    seqM = DDMC(i, ncl=30, SeqWeight=1e6, distance_method="Binomial", random_state=5).fit(d)
    models = [dataM, mixM, seqM]

    # Center to peptide distance
    barplot_PeptideToClusterDistances(models, ax[1], n=2000)

    # Position Enrichment
    boxplot_TotalPositionEnrichment(models, ax[2])

    return f


def calculate_AUCs_phenotypes(ax, X, nRuns=3, ncl=35):
    """Plot mean AUCs per phenotype across weights."""
    # Signaling
    d = X.select_dtypes(include=[float]).T
    i = X.select_dtypes(include=[object])

    # Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)

    # LASSO
    lr = LogisticRegressionCV(cv=3, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")

    weights = [0, 50, 100, 250, 500, 750, 1000, 1000000]
    run, ws, stk, ea, hcb = [], [], [], [], []
    for w in weights:
        for r in range(nRuns):
            run.append(r)
            ws.append(w)
            model = DDMC(i, ncl=ncl, SeqWeight=w, distance_method="Binomial").fit(d)

            # Find and scale centers
            centers_gen, centers_hcb = TransformCenters(model, X)

            # STK11
            stk.append(plotROC(ax, lr, centers_gen.values, y["STK11.mutation.status"], cv_folds=3, return_mAUC=True, kfold="Repeated"))

            # EGFRm/ALKf
            y_EA = merge_binary_vectors(y.copy(), "EGFR.mutation.status", "ALK.fusion")
            ea.append(plotROC(ax, lr, centers_gen.values, y_EA, cv_folds=3, return_mAUC=True, kfold="Repeated"))

            # Hot-Cold behavior
            y_hcb, centers_hcb = HotColdBehavior(centers_hcb)
            hcb.append(plotROC(ax, lr, centers_hcb.values, y_hcb, cv_folds=3, return_mAUC=True, kfold="Repeated"))

    out = pd.DataFrame()
    out["Run"] = run
    out["Weight"] = ws
    out["STK11"] = stk
    out["EGFRm/ALKf"] = ea
    out["Infiltration"] = hcb

    return out


def barplot_PeptideToClusterDistances(models, ax, n=3000):
    """Compute and plot p-signal-to-center and motif to cluster distance for n peptides across weights."""
    # Import signaling data, select random peptides, and find cluster assignments
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, tmt=2)
    random_peptides = np.random.choice(list(np.arange(len(models[0].labels()))), n, replace=False)
    X["labels0"] = models[0].labels()
    X["labels500"] = models[1].labels()
    X["labels1M"] = models[2].labels()
    X = X.iloc[random_peptides, :]
    labels = X.loc[:, ["labels0", "labels500", "labels1M"]].values.T
    d = X.select_dtypes(include=[float]).values

    psDist = np.zeros((3, d.shape[0]))
    for ii, model in enumerate(models):
        for jj in range(d.shape[0]):
            # Data distance
            center = model.transform()[:, labels[ii, jj] - 1]
            idx_values = np.argwhere(~np.isnan(d[jj, :]))
            psDist[ii, jj] = mean_squared_error(d[jj, idx_values], center[idx_values])

    psDist = pd.DataFrame(psDist).T
    psDist.columns = ["p-Abundance", "Mix", "Sequence"]
    ps_data = pd.melt(psDist, value_vars=["p-Abundance", "Mix", "Sequence"], var_name="Model", value_name="p-signal MSE")
    sns.barplot(data=ps_data, x="Model", y="p-signal MSE", ci=None, ax=ax)
    ax.set_title("Peptide-to-Cluster signal MSE")


def boxplot_TotalPositionEnrichment(models, ax):
    """Position enrichment of cluster PSSMs"""
    enr = np.zeros((3, models[0].n_components), dtype=float)
    for ii, model in enumerate(models):
        pssms = model.pssms()[0]
        for jj in range(len(pssms)):
            enr[ii, jj] = np.sum(pssms[jj].sum().drop(5))

    enr = pd.DataFrame(enr)
    enr.columns = np.arange(models[0].n_components) + 1
    enr.insert(0, "Model", ["p-Abundance", "Mix", "Sequence"])
    dd = pd.melt(frame=enr, id_vars="Model", value_vars=enr.columns[1:], var_name="Cluster", value_name="Total information (bits)")
    sns.stripplot(data=dd, x="Model", y="Total information (bits)", ax=ax)
    sns.boxplot(data=dd, x="Model", y="Total information (bits)", ax=ax, fliersize=0)
    ax.set_title("Cumulative PSSM Enrichment")
