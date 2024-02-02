import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from ddmc.clustering import DDMC
from ddmc.figures.common import getSetup
from ddmc.logistic_regression import plot_roc
from ddmc.datasets import CPTAC, filter_incomplete_peptides


def makeFigure():
    axes, f = getSetup((9, 6), (2, 3), multz={3: 2})

    # Import data
    cptac = CPTAC()
    p_signal = filter_incomplete_peptides(cptac.get_p_signal(), sample_presence_ratio=1)
    is_tumor = cptac.get_tumor_or_nat(p_signal.columns)

    n_components = 30
    model_gmm = DDMC(
        n_components=n_components,
        seq_weight=0,
        distance_method="Binomial",
        random_state=5,
    ).fit(p_signal)

    model_kmeans = KMeans(n_clusters=n_components).fit(p_signal.values)

    lr = LogisticRegressionCV(
        cv=3,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="elasticnet",
        l1_ratios=[0.85],
        class_weight="balanced",
    )

    plot_roc(lr, p_signal.T.values, is_tumor, ax=axes[0])
    axes[0].set_title("unclustered ROC")

    # plot regression coefficients
    coefs = pd.Series(lr.coef_.squeeze(), index=p_signal.index)
    coefs.sort_values(ascending=False, inplace=True)
    coefs.dropna(inplace=True)
    coefs = pd.concat([coefs.iloc[:10], coefs.iloc[-10:]])
    coefs = coefs.to_frame("Coefficient")
    coefs.reset_index(names=["p-site"], inplace=True)
    sns.barplot(data=coefs, x="p-site", y="Coefficient", color="darkblue", ax=axes[3])
    axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=45)

    plot_roc(lr, model_gmm.transform(), is_tumor, ax=axes[1])
    axes[1].set_title("GMM ROC")

    plot_roc(lr, model_kmeans.cluster_centers_.T, is_tumor, ax=axes[2])
    axes[2].set_title("kmeans roc")

    return f
