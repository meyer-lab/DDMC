"""Logistic Regression Model functions to predict clinical features of CPTAC patients given their clustered phosphoproteomes."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold


def plotClusterCoefficients(ax, lr):
    """Plot LR coeficients of clusters."""
    coefs_ = pd.DataFrame(lr.coef_.T, columns=["Coefficient"])
    coefs_["Cluster"] = np.arange(coefs_.shape[0]) + 1
    sns.barplot(x="Cluster", y="Coefficient", data=coefs_)
    ax.set_title("Logistic Regression Cluster Coefficients")


def plotPredictionProbabilities(ax, lr, y_pred, dd, yy):
    """Plot LR predictions and prediction probabilities."""
    res_ = pd.DataFrame()
    res_["y, p(x)"] = lr.predict_proba(dd)[:, 1]
    z = y_pred == yy
    res_["Correct_Prediction"] = z.values
    res_["Prediction"] = lr.predict(dd).astype("int")
    res_["Patients"] = np.arange(res_.shape[0]) + 1
    sns.scatterplot(x="Patients", y="Prediction", data=res_, hue="Correct_Prediction")
    sns.lineplot(x="Patients", y="y, p(x)", data=res_, marker="s", color="gray")
    ax.axhline(0.5, ls='--', color='r')


def plotConfusionMatrix(ax, lr, dd, yy):
    """Actual vs predicted outputs"""
    cm = confusion_matrix(yy, lr.predict(dd))
    n = lr.classes_.shape[0]
    ax.imshow(cm)
    ax.grid(False)
    ax.set_xlabel('Predicted outputs', color='black')
    ax.set_ylabel('Actual outputs', color='black')
    ax.xaxis.set(ticks=range(n))
    ax.yaxis.set(ticks=range(n))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')


def plotROC(ax, classifier, d, y, cv_folds=4):
    """Plot Receiver Operating Characteristc with cross-validation folds of a given classifier model. 
    Note that it doesn't need to be a logistic regression."""
    d = d.values
    y = y.values
    cv = StratifiedKFold(n_splits=cv_folds)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv.split(d, y)):
        classifier.fit(d[train], y[train])
        viz = plot_roc_curve(classifier, d[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver Operating Characteristic")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2)
