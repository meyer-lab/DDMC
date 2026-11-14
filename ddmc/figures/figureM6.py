from sklearn.linear_model import LogisticRegressionCV

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC
from ddmc.figures.common import (
    getSetup,
    plot_cluster_kinase_distances,
    plot_p_signal_across_clusters_and_binary_feature,
)
from ddmc.logistic_regression import (
    get_highest_weighted_clusters,
    normalize_cluster_centers,
    plot_cluster_regression_coefficients,
    plot_roc,
)


def makeFigure():
    axes, f = getSetup((11, 7), (2, 3), multz={0: 1})
    cptac = CPTAC()
    p_signal = cptac.get_p_signal()
    model = DDMC(n_components=30, seq_weight=100).fit(p_signal)

    # Import Genotype data
    egfrm = cptac.get_mutations(["EGFR.mutation.status"])["EGFR.mutation.status"]

    # Find centers
    centers = model.transform(as_df=True).loc[egfrm.index]

    plot_p_signal_across_clusters_and_binary_feature(
        egfrm, centers, "egfr mutation", axes[0]
    )

    # Normalize
    centers.iloc[:, :] = normalize_cluster_centers(centers.values)

    # Logistic Regression
    lr = LogisticRegressionCV(
        cv=20,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )
    plot_roc(
        lr, centers.values, egfrm.values, cv_folds=3, title="ROC EGFRm", ax=axes[1]
    )
    axes[1].legend(loc="lower right", prop={"size": 8})

    plot_cluster_regression_coefficients(axes[2], lr, title="")

    top_clusters = get_highest_weighted_clusters(model, lr.coef_)

    distances = model.predict_upstream_kinases()[top_clusters]

    # plot Upstream Kinases
    plot_cluster_kinase_distances(
        distances, model.get_pssms(clusters=top_clusters), axes[3], num_hits=2
    )
    return f
