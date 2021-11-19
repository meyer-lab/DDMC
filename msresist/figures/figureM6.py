"""
This creates Figure 6: STK11 analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from bioinfokit import visuz
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides
from .figure2 import plotDistanceToUpstreamKinase
from .figureM4 import find_patients_with_NATandTumor
from .figureM5 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals, plot_enriched_processes
from ..logistic_regression import plotROC, plotClusterCoefficients
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((11, 7), (2, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
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
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.n_components) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.set_index("Patient_ID")
    centers = centers.drop([14, 24], axis=1)  # Drop clusters 14&24, contain only 1 peptide

    # Hypothesis Testing
    assert np.all(y['Sample.ID'] == centers.index)
    centers["EGFRm"] = y["EGFR.mutation.status"].values
    pvals = calculate_mannW_pvals(centers, "EGFRm", 1, 0)
    pvals = build_pval_matrix(model.n_components, pvals)
    centers["EGFRm"] = centers["EGFRm"].replace(0, "EGFR WT")
    centers["EGFRm"] = centers["EGFRm"].replace(1, "EGFRm")
    plot_clusters_binaryfeatures(centers, "EGFRm", ax[0], pvals=pvals)
    ax[0].legend(loc='lower left', prop={'size': 10})

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    centers = centers.reset_index().set_index("EGFRm")
    centers = find_patients_with_NATandTumor(centers, "Patient_ID", conc=False)
    y_ = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y_.index.values), "Samples don't match"

    # Normalize
    centers = centers.T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T

    # Logistic Regression
    centers["EGFRm"] = y_["EGFR.mutation.status"].values
    lr = LogisticRegressionCV(cv=20, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[1], lr, centers.iloc[:, :-1].values, centers["EGFRm"], cv_folds=4, title="ROC EGFRm")
    ax[1].legend(loc='lower right', prop={'size': 8})
    plotClusterCoefficients(ax[2], lr.fit(centers.iloc[:, :-1].values, centers["EGFRm"]), title="")
    ax[2].legend(loc='lower left', prop={'size': 10})

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [5, 16, 27], ax[3], num_hits=3)

    # volcano protein expression
    ax[-1].axis("off")

    return f


def make_EGFRvolcano_plot(centers, y):
    """ Make volcano plot with differential protein expression between EGFRm and WT. """
    y = y.drop("C3N.00545")
    centers = centers.drop("C3N.00545")
    prot = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_LUAD_Protein.csv").iloc[:, 15:].dropna().drop("id.1", axis=1).drop_duplicates()
    prot = prot.set_index("GeneSymbol").T.sort_index().reset_index()
    prot = prot[~prot["index"].str.endswith(".N")].set_index("index")
    prot.columns.name = None

    l1 = list(centers.index)
    l2 = list(prot.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    prot = prot.drop(dif)
    assert np.all(centers.index.values == prot.index.values), "Samples don't match"

    egfr = y["EGFR.mutation.status"].replace(0, "WT")
    egfr = egfr.replace(1, "Mut")
    prot["EGFR Status"] = egfr.values
    prot = prot.set_index("EGFR Status")

    pvals = mannwhitneyu(prot.loc["Mut"], prot.loc["WT"])[1]
    pvals = multipletests(pvals)[1]

    means = prot.reset_index().groupby("EGFR Status").mean()
    fc = means.iloc[0, :] - means.iloc[1, :]

    pv = pd.DataFrame()
    pv["Gene"] = prot.columns
    pv["p-values"] = pvals
    pv["logFC"] = fc.values
    pv = pv.sort_values(by="p-values")

    visuz.gene_exp.volcano(df=pv, lfc='logFC', pv='p-values', show=True, geneid="Gene", genenames='deg', figtype="svg")
