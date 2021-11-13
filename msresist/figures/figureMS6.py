"""
This creates Figure 6: STK11 analysis
"""
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides
from .figure2 import plotDistanceToUpstreamKinase
from .figureM4 import find_patients_with_NATandTumor
from .figureM5 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from ..logistic_regression import plotROC, plotClusterCoefficients
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((11, 7), (2, 3), multz={0: 1, 3: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Phosphoproteomic aberrations associated with molecular signatures
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import signaling data
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=2)
    d = X.select_dtypes(include=[float]).T
    i = X.select_dtypes(include=[object])

    # Fit DDMC
    model = MassSpecClustering(i, ncl=30, SeqWeight=100, distance_method="Binomial", random_state=5).fit(d)

    # Import Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]

    # Find centers
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.n_components) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.set_index("Patient_ID")
    centers = centers.drop([14, 24], axis=1)  # Drop clusters 14&24, contain only 1 peptide

    # Hypothesis Testing
    assert np.all(y['Sample.ID'] == centers.index)
    centers["STK11"] = y["STK11.mutation.status"].values
    pvals = calculate_mannW_pvals(centers, "STK11", 1, 0)
    pvals = build_pval_matrix(model.n_components, pvals)
    centers["STK11"] = centers["STK11"].replace(0, "STK11 WT")
    centers["STK11"] = centers["STK11"].replace(1, "STK11m")
    plot_clusters_binaryfeatures(centers, "STK11", ax[0], pvals=pvals)
    ax[0].legend(loc='lower left', prop={'size': 10})

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    centers = centers.reset_index().set_index("STK11")
    centers = find_patients_with_NATandTumor(centers, "Patient_ID", conc=True)
    y_ = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y_.index.values), "Samples don't match"

    # Normalize
    centers = centers.T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T

    # Logistic Regression
    centers["STK11"] = y_["STK11.mutation.status"].values
    lr = LogisticRegressionCV(cv=15, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[1], lr, centers.iloc[:, :-1].values, centers["STK11"], cv_folds=4, title="ROC STK11")
    ax[1].legend(loc='lower right', prop={'size': 8})
    plotClusterCoefficients(ax[2], lr.fit(centers.iloc[:, :-1], centers["STK11"].values), list(centers.columns[:-1]), title="")
    ax[2].legend(loc='lower left', prop={'size': 10})

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [16, 18], ax[3], num_hits=3, PsP_background=False)

    return f
