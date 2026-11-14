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
    axes, f = getSetup((11, 7), (2, 3), multz={0: 1, 4: 1})
    cptac = CPTAC()
    is_hot = cptac.get_hot_cold_labels()
    p_signal = cptac.get_p_signal()
    model = DDMC(n_components=30, seq_weight=100, random_state=5).fit(p_signal)

    centers = model.transform(as_df=True).loc[is_hot.index]
    centers.iloc[:, :] = normalize_cluster_centers(centers.values)

    plot_p_signal_across_clusters_and_binary_feature(is_hot, centers, "is_hot", axes[0])

    n_cv = 15
    lr = LogisticRegressionCV(
        cv=n_cv, solver="saga", n_jobs=1, penalty="l1", max_iter=10000
    )
    plot_roc(
        lr,
        centers.values,
        is_hot.values,
        cv_folds=n_cv,
        title="ROC TI",
        return_mAUC=True,
    )
    plot_cluster_regression_coefficients(axes[1], lr, title="")

    top_clusters = get_highest_weighted_clusters(model, lr.coef_)

    distances = model.predict_upstream_kinases()[top_clusters]

    plot_cluster_kinase_distances(
        distances, model.get_pssms(clusters=top_clusters), axes[2], num_hits=2
    )
    return f
