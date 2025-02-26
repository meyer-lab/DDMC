"""Logistic Regression Model functions to predict clinical features of CPTAC patients given their clustered phosphoproteomes."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import sem
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from ddmc.clustering import DDMC


def normalize_cluster_centers(centers: np.ndarray):
    # normalize centers along along patient dimension
    return StandardScaler(with_std=False).fit_transform(centers.T).T


def get_highest_weighted_clusters(model: DDMC, coefficients: np.ndarray, n_clusters=3):
    top_clusters = np.flip(np.argsort(np.abs(coefficients.squeeze())))
    top_clusters = [
        cluster for cluster in top_clusters if cluster in model.get_nonempty_clusters()
    ]
    return top_clusters[:n_clusters]


def plot_cluster_regression_coefficients(ax: Axes, lr, hue=None, title=False):
    """Plot LR coeficients of clusters."""
    coefs_ = pd.DataFrame(lr.coef_.T, columns=["LR Coefficient"])
    if hue:
        coefs_["Cluster"] = [l.split("_")[0] for l in hue]
        coefs_["Sample"] = [l.split("_")[1] for l in hue]
        hue = "Sample"
    else:
        coefs_["Cluster"] = np.arange(coefs_.shape[0])
    p = sns.barplot(
        ax=ax,
        x="Cluster",
        y="LR Coefficient",
        hue=hue,
        data=coefs_,
        color="darkblue",
        **{"linewidth": 0.5},
        **{"edgecolor": "black"}
    )

    p.tick_params(axis="x", labelsize=6)
    if title:
        ax.set_title(title)


def plot_roc(
    classifier,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 4,
    title=False,
    return_mAUC: bool = False,
    kfold="Stratified",
    ax: Axes = None,
):
    """Plot Receiver Operating Characteristc with cross-validation folds of a given classifier model."""
    if kfold == "Stratified":
        cv = StratifiedKFold(n_splits=cv_folds)
    elif kfold == "Repeated":
        cv = RepeatedKFold(n_splits=cv_folds, n_repeats=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train], y.iloc[train])
        viz = RocCurveDisplay.from_estimator(classifier, X.iloc[test], y.iloc[test])
        plt.close()
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    if return_mAUC:
        return mean_auc

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    sem_auc = sem(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, sem_auc),
        lw=2,
        alpha=0.8,
    )

    sem_tpr = sem(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + sem_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - sem_tpr, 0)
    ax.fill_between(
        mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 SEM"
    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC")

    if title:
        ax.set_title(title)

    ax.legend(loc=4, prop={"size": 8}, labelspacing=0.2)
