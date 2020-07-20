"""Logistic Regression Model functions to predict clinical features of CPTAC patients given their clustered phosphoproteomes."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

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
