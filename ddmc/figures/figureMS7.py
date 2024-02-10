import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, select_peptide_subset
from ddmc.figures.common import (
    plot_cluster_kinase_distances,
    plot_p_signal_across_clusters_and_binary_feature,
    getSetup,
)
from ddmc.logistic_regression import (
    plot_roc,
    plot_cluster_regression_coefficients,
    normalize_cluster_centers,
    get_highest_weighted_clusters,
)


def makeFigure():
    axes, f = getSetup((11, 7), (2, 3), multz={0: 1})
    cptac = CPTAC()
    p_signal = cptac.get_p_signal()
    stk11m = cptac.get_mutations(["STK11.mutation.status"])["STK11.mutation.status"]

    model = DDMC(n_components=30, seq_weight=100, random_state=5).fit(p_signal)
    centers = model.transform(as_df=True)
    centers = centers.loc[stk11m.index]
    plot_p_signal_across_clusters_and_binary_feature(
        stk11m.values, centers, "STK11M", axes[0]
    )

    centers.iloc[:, :] = normalize_cluster_centers(centers.values)

    # Logistic Regression
    lr = LogisticRegressionCV(
        cv=5,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
        random_state=10,
    )

    plot_roc(
        lr, centers.values, stk11m.values, cv_folds=4, title="ROC STK11", ax=axes[1]
    )

    plot_cluster_regression_coefficients(
        axes[2],
        lr,
    )

    top_clusters = get_highest_weighted_clusters(model, lr.coef_)

    plot_cluster_kinase_distances(
        model.predict_upstream_kinases()[top_clusters],
        model.get_pssms(PsP_background=True, clusters=top_clusters)[0],
        ax=axes[3],
    )
    return f
