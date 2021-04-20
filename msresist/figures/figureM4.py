"""
This creates Figure 3: Tumor vs NAT analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from .common import subplotLabel, getSetup
from .figureMS6 import TumorType
from ..logistic_regression import plotClusterCoefficients, plotROC
from ..figures.figure2 import plotPCA, plotMotifs, plotDistanceToUpstreamKinase
from ..pre_processing import MeanCenter


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((13, 10), (3, 3), multz={0: 1, 4: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    import pandas as pd
    from msresist.clustering import MassSpecClustering
    from msresist.validations import preprocess_ebdt_mcf7

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # first plot heatmap of clusters
    ax[0].axis("off")

    # Normalize
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = TumorType(centers).set_index("Patient_ID")
    centers["Type"] = centers["Type"].replace("Normal", "NAT")

    # PCA and Hypothesis Testing
    pvals = calculate_mannW_pvals(centers, "Type", "NAT", "Tumor")
    pvals = build_pval_matrix(model.ncl, pvals)
    plotPCA(ax[1:3], centers.reset_index(), 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", pvals=pvals.iloc[:, -1].values)
    plot_clusters_binaryfeatures(centers, "Type", ["Tumor", "NAT"], ax[3], pvals=pvals)

    # Transform to Binary
    c = centers.select_dtypes(include=['float64'])
    tt = centers.iloc[:, -1]
    tt = tt.replace("NAT", 0)
    tt = tt.replace("Tumor", 1)

    # Logistic Regression
    lr = LogisticRegressionCV(Cs=10, cv=10, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[4], lr, c.values, tt, cv_folds=4)
    plotClusterCoefficients(ax[5], lr)

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [11, 12, 23], ax[6], num_hits=3)

    return f


def plot_clusters_binaryfeatures(centers, id_var, labels, ax, pvals=False):
    """Plot p-signal of binary features (tumor vs NAT or mutational status) per cluster """
    ncl = centers.shape[1] - 1
    data = pd.melt(id_vars=id_var, value_vars=np.arange(ncl) + 1, value_name="p-signal", var_name="Cluster", frame=centers)
    sns.stripplot(x="Cluster", y="p-signal", hue=id_var, data=data, dodge=True, ax=ax, alpha=0.2)
    sns.boxplot(x="Cluster", y="p-signal", hue=id_var, data=data, dodge=True, ax=ax, color="white", linewidth=2)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(title=id_var, labels=labels, handles=handles[2:], prop={'size': 8})
    if not isinstance(pvals, bool):
        for ii, s in enumerate(pvals["Significant"]):
            y, h, col = data['p-signal'].max(), .05, 'k'
            if s == "Not Significant":
                continue
            elif s == "<0.05":
                mark = "*"
            else:
                mark = "**"
            ax.text(ii, y + h, mark, ha='center', va='bottom', color=col, fontsize=20)


def calculate_mannW_pvals(centers, col, feature1, feature2):
    """Compute Mann Whitney rank test p-values"""
    pvals = []
    for ii in range(centers.shape[1] - 1):
        x = centers.iloc[:, [ii, -1]]
        x1 = x[x[col] == feature1].iloc[:, 0]
        x2 = x[x[col] == feature2].iloc[:, 0]
        pval = mannwhitneyu(x1, x2)[1]
        pvals.append(pval)
    pvals = multipletests(pvals)[1]  # p-value correction for multiple tests
    return pvals


def build_pval_matrix(ncl, pvals):
    """Build data frame with pvalues per cluster"""
    data = pd.DataFrame()
    data["Clusters"] = np.arange(ncl) + 1
    data["p-value"] = pvals
    signif = []
    for val in pvals:
        if 0.01 < val < 0.05:
            signif.append("<0.05")
        elif val < 0.01:
            signif.append("<0.01")
        else:
            signif.append("Not Significant")
    data["Significant"] = signif
    return data
