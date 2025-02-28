from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, select_peptide_subset
from ddmc.figures.common import getSetup
from ddmc.logistic_regression import plot_roc, normalize_cluster_centers


def makeFigure():
    cptac = CPTAC()

    mutations = cptac.get_mutations(
        ["STK11.mutation.status", "EGFR.mutation.status", "ALK.fusion"]
    )
    stk11 = mutations["STK11.mutation.status"]
    egfr = mutations["EGFR.mutation.status"]
    hot_cold = cptac.get_hot_cold_labels()

    p_signal = CPTAC().get_p_signal()

    # LASSO
    lr = LogisticRegressionCV(
        Cs=10,
        cv=10,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )

    folds = 3
    weights = [0, 500, 1000000]
    ax, f = getSetup((15, 10), (3, len(weights)))
    for ii, weight in enumerate(weights):
        model = DDMC(
            n_components=30,
            seq_weight=weight,
            distance_method="Binomial",
            random_state=5,
        ).fit(p_signal)

        # Find and scale centers
        centers = model.transform(as_df=True)
        centers.iloc[:, :] = normalize_cluster_centers(centers.values)

        # STK11
        plot_roc(
            lr,
            centers.loc[stk11.index].values,
            stk11,
            cv_folds=folds,
            return_mAUC=False,
            ax=ax[ii],
            title="STK11m " + "w=" + str(model.seq_weight),
        )
        plot_roc(
            lr,
            centers.loc[egfr.index].values,
            egfr,
            cv_folds=folds,
            return_mAUC=False,
            ax=ax[ii + len(weights)],
            title="EGFRm " + "w=" + str(model.seq_weight),
        )
        plot_roc(
            lr,
            centers.loc[hot_cold.index].values,
            hot_cold,
            cv_folds=folds,
            return_mAUC=False,
            ax=ax[ii + len(weights) * 2],
            title="Infiltration " + "w=" + str(model.seq_weight),
        )

    return f
