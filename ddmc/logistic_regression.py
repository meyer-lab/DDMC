"""Logistic Regression Model functions to predict clinical features of CPTAC patients given their clustered phosphoproteomes.

Contains:
    - `normalize_cluster_centers`: mean-centers `DDMC` cluster centers along
      the patient dimension, for use as classifier features.
    - `get_highest_weighted_clusters`: picks out the clusters a fitted
      classifier weighted most heavily.
    - `plot_cluster_regression_coefficients` / `plot_roc`: plotting helpers
      for a classifier's per-cluster coefficients and its cross-validated
      ROC curve.
"""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import sem
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ddmc.clustering import DDMC


def normalize_cluster_centers(centers: np.ndarray) -> np.ndarray:
    """Mean-center cluster centers along the patient/sample dimension.

    Args:
        centers: Cluster centers of shape (n_samples, n_components), e.g.
            from `DDMC.transform()`.

    Returns:
        `centers` with each cluster's (column's) values shifted to have
        zero mean across samples, same shape as `centers`.
    """
    # normalize centers along along patient dimension
    return StandardScaler(with_std=False).fit_transform(centers)


def get_highest_weighted_clusters(
    model: DDMC, coefficients: np.ndarray, n_clusters: int = 3
) -> list[int]:
    """Pick out the (nonempty) clusters a fitted classifier weighted most heavily.

    Args:
        model: The fitted `DDMC` model the classifier's features came from
            (used to exclude empty clusters).
        coefficients: Per-cluster classifier coefficients, e.g.
            `lr.coef_`, of shape (1, n_components) or (n_components,).
        n_clusters: Maximum number of top clusters to return.

    Returns:
        Up to `n_clusters` nonempty cluster indices, ordered by decreasing
        absolute coefficient magnitude.
    """
    top_clusters = np.flip(np.argsort(np.abs(coefficients.squeeze())))
    top_clusters = [
        cluster for cluster in top_clusters if cluster in model.get_nonempty_clusters()
    ]
    return top_clusters[:n_clusters]


def plot_cluster_regression_coefficients(
    ax: Axes, lr: Any, hue: Sequence[str] | None = None, title=False
) -> None:
    """Plot LR coeficients of clusters.

    Args:
        ax: Axes to plot onto.
        lr: A fitted scikit-learn linear classifier exposing `coef_` of
            shape (1, n_components).
        hue: If given, per-cluster-run labels formatted as
            `"{cluster}_{sample}"` (split on `"_"`) to group/color bars by
            sample when coefficients from multiple runs are concatenated;
            not used by any current figure (all call `plot_roc` with the
            default `hue=None`, one bar per cluster).
        title (str | bool): If given (and not `False`), set as the axes title.
    """
    coefs_ = pd.DataFrame(lr.coef_.T, columns=["LR Coefficient"])
    if hue:
        coefs_["Cluster"] = [label.split("_")[0] for label in hue]
        coefs_["Sample"] = [label.split("_")[1] for label in hue]
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
        **{"edgecolor": "black"},
    )

    p.tick_params(axis="x", labelsize=6)
    if title:
        ax.set_title(title)


def plot_roc(
    classifier: Any,
    X: np.ndarray,
    y: np.ndarray | pd.Series,
    cv_folds: int = 4,
    title=False,
    return_mAUC: bool = False,
    kfold: str = "Stratified",
    ax: Axes | None = None,
) -> float | None:
    """Plot Receiver Operating Characteristc with cross-validation folds of a given classifier model.

    Fits a fresh copy of `classifier` on each cross-validation fold, plots
    the mean ROC curve (+/- 1 SEM band) across folds, and optionally
    returns just the mean AUC instead of plotting.

    Args:
        classifier: A scikit-learn-compatible classifier exposing `fit`.
        X: Feature matrix of shape (n_samples, n_features).
        y: Binary target labels of shape (n_samples,).
        cv_folds: Number of cross-validation folds.
        title (str | bool): If given (and not `False`), set as the axes title.
        return_mAUC: If True, skip plotting and just return the mean AUC.
        kfold: Cross-validation strategy: `"Stratified"`
            (`StratifiedKFold`) or `"Repeated"` (`RepeatedKFold`, 10
            repeats).
        ax: Axes to plot onto; defaults to the current axes (`plt.gca()`).

    Returns:
        The mean AUC across folds if `return_mAUC` is True, else `None`
        (the ROC curve is plotted onto `ax` instead).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if kfold == "Stratified":
        cv = StratifiedKFold(n_splits=cv_folds)
    elif kfold == "Repeated":
        cv = RepeatedKFold(n_splits=cv_folds, n_repeats=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(classifier, X[test], y[test])
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

    if ax is None:
        ax = plt.gca()

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    sem_auc = sem(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=rf"Mean ROC (AUC = {mean_auc:0.2f} $\pm$ {sem_auc:0.2f})",
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
