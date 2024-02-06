from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV

from ddmc.clustering import DDMC
from ddmc.figures.common import getSetup
from ddmc.logistic_regression import plot_roc, normalize_cluster_centers
from ddmc.datasets import CPTAC, select_peptide_subset


def makeFigure():
    axes, f = getSetup((5, 5), (2, 2), multz={0: 1})

    # use small numbers here so it doesn't take forever
    regression_results = do_phenotype_regression(n_runs=1, n_cv_folds=2)
    plot_phenotype_regression(regression_results, axes[0])
    p_signal = select_peptide_subset(CPTAC().get_p_signal(), keep_num=2000)
    models = [
        DDMC(
            n_components=30,
            seq_weight=0,
            distance_method="Binomial",
            random_state=5,
        ).fit(p_signal),
        DDMC(
            n_components=30,
            seq_weight=250,
            distance_method="Binomial",
            random_state=5,
        ).fit(p_signal),
        DDMC(
            n_components=30,
            seq_weight=1e6,
            distance_method="Binomial",
            random_state=5,
        ).fit(p_signal),
    ]
    plot_peptide_to_cluster_p_signal_distances(p_signal, models, axes[1])
    plot_total_position_enrichment(models, axes[2])
    return f


def do_phenotype_regression(n_runs=3, n_components=35, n_cv_folds=3):
    """Plot mean AUCs per phenotype across weights."""
    cptac = CPTAC()
    p_signal = cptac.get_p_signal()

    mutations = cptac.get_mutations(
        ["STK11.mutation.status", "EGFR.mutation.status", "ALK.fusion"]
    )
    stk11 = mutations["STK11.mutation.status"]
    egfr = mutations["EGFR.mutation.status"]
    alk = mutations["ALK.fusion"]
    egfr_or_alk = egfr | alk
    hot_cold = cptac.get_hot_cold_labels()

    lr = LogisticRegressionCV(
        cv=3,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )

    results = pd.DataFrame(columns=["Weight", "STK11", "EGFRm/ALKf", "Infiltration"])

    for seq_weight in [0, 1e2, 1e6]:
        for _ in range(n_runs):
            ddmc = DDMC(
                n_components=n_components,
                seq_weight=seq_weight,
                distance_method="Binomial",
            ).fit(p_signal)
            centers = ddmc.transform(as_df=True)
            centers.iloc[:, :] = normalize_cluster_centers(centers.values)
            # the available patients vary by label
            stk11_auc = plot_roc(
                lr,
                centers.loc[stk11.index].values,
                stk11,
                cv_folds=n_cv_folds,
                return_mAUC=True,
                kfold="Repeated",
            )
            egfr_or_alk_auc = plot_roc(
                lr,
                centers.loc[egfr_or_alk.index].values,
                egfr_or_alk,
                cv_folds=n_cv_folds,
                return_mAUC=True,
                kfold="Repeated",
            )
            hot_cold_auc = plot_roc(
                lr,
                centers.loc[hot_cold.index].values,
                hot_cold,
                cv_folds=n_cv_folds,
                return_mAUC=True,
                kfold="Repeated",
            )
            results.loc[len(results)] = [
                seq_weight,
                stk11_auc,
                egfr_or_alk_auc,
                hot_cold_auc,
            ]

    return results


def plot_phenotype_regression(results: pd.DataFrame, ax) -> None:
    df_melted = results.melt(
        id_vars=["Weight"], var_name="Task", value_name="Performance"
    )
    gmm = df_melted[df_melted["Weight"] == 0].copy()
    ddmc = df_melted[df_melted["Weight"] != 0].copy()
    best_ddmc = ddmc.loc[ddmc.groupby("Task")["Performance"].idxmax()]
    gmm["Type"] = "GMM"
    best_ddmc["Type"] = "DDMC"
    plot_data = pd.concat([gmm, best_ddmc])
    sns.barplot(data=plot_data, x="Task", y="Performance", hue="Type", ax=ax)
    ax.set_title("Performance by Task and Weight Type")
    ax.set_xlabel("Performance")
    ax.set_ylabel("Regression Task")


def plot_peptide_to_cluster_p_signal_distances(
    p_signal: pd.DataFrame, models: List[DDMC], ax, n_peptides=100
):
    peptide_idx = np.random.choice(len(p_signal), n_peptides)
    seq_weights = [model.seq_weight for model in models]
    dists = pd.DataFrame(
        np.zeros((n_peptides, len(models)), dtype=float), columns=seq_weights
    )
    for i, model in enumerate(models):
        labels = model.labels()
        centers = model.transform().T
        dists.iloc[:, i] = np.nanmean(
            (p_signal.iloc[peptide_idx] - centers[labels[peptide_idx]]) ** 2, axis=1
        )
    dists_melted = pd.melt(
        dists,
        value_vars=seq_weights,
        var_name="Sequence Weight",
        value_name="p-signal MSE",
    )
    sns.barplot(dists_melted, x="Sequence Weight", y="p-signal MSE", ax=ax)
    ax.set_title("Peptide-to-cluster p-signal MSE")


def plot_total_position_enrichment(models: List[DDMC], ax):
    """Position enrichment of cluster PSSMs"""
    enrichment = pd.DataFrame(
        columns=["Sequence Weight", "Component", "Total information (bits)"]
    )

    # loop because it's not guaranteed that each cluster will contain a peptide
    for model in models:
        pssm_names, pssms = model.get_pssms()
        for cluster, pssm in zip(pssm_names, pssms):
            enrichment.loc[len(enrichment)] = [
                model.seq_weight,
                cluster,
                np.sum(np.delete(pssm, 5, axis=1)),
            ]

    sns.stripplot(enrichment, x="Sequence Weight", y="Total information (bits)", ax=ax)
    sns.boxplot(
        enrichment,
        x="Sequence Weight",
        y="Total information (bits)",
        ax=ax,
        fliersize=0,
    )
    ax.set_title("Cumulative PSSM Enrichment")
