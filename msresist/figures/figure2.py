"""
This creates Figure 2.
"""

import os
import pandas as pd
import numpy as np
from .common import subplotLabel, getSetup
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd
import math
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from msresist.parameter_tuning import kmeansPLSR_tuning
from msresist.plsr import Q2Y_across_components, R2Y_across_components
from msresist.clustering import MyOwnKMEANS
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from msresist.pre_processing import preprocessing, MergeDfbyMean


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 9), (2, 2))

    # blank out first axis for cartoon
#     ax[0].axis('off')

    #Cell Viability
    Y_cv1 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw3.csv')).iloc[:30, :11]
    Y_cv2 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/CV_raw4.csv')).iloc[:29, :11]

    for ii in range(1, Y_cv2.columns.size):
        Y_cv1.iloc[:, ii] /= Y_cv1.iloc[0, ii]
        Y_cv2.iloc[:, ii] /= Y_cv2.iloc[0, ii]

    Y_cv = MergeDfbyMean(pd.concat([Y_cv1, Y_cv2], axis=0), Y_cv1.columns, "Elapsed")
    Y_cv = Y_cv.reset_index()[Y_cv1.columns]
    Y_cv = Y_cv[Y_cv["Elapsed"] == 72].iloc[0, 1:]

    #Phosphorylation data
    ABC_mc = preprocessing(motifs=True, Vfilter=False, FCfilter=True, log2T=True)

    header = ABC_mc.columns
    treatments = ABC_mc.columns[2:]

    data = ABC_mc.iloc[:,2:].T

    #Set up model pipeline
    ncl, ncomp = 4, 2
    estimators = [('kmeans', MyOwnKMEANS(ncl)), ('plsr', PLSRegression(ncomp))]
    kmeans_plsr = Pipeline(estimators)

    plotR2YQ2Y(axs[0], ncl, data, Y_cv)

    plotGridSearch(axs[1], data ,Y_cv)

    plotActualvsPred(axs[2], kmeans_plsr, data, Y_cv)

    plotScoresLoadings(axs[3:5], kmeans_plsr, data, Y_cv, ncl)

    # Add subplot labels
    subplotLabel(ax)

    return f

def plotR2YQ2Y(axs, ncl, X, Y):
    kmeans = MyOwnKMEANS(ncl).fit(X, Y)
    centers = kmeans.transform(X)

    maxComp = ncl
    Q2Y = Q2Y_across_components(centers, Y_cv, maxComp+1)
    R2Y = R2Y_across_components(centers, Y_cv, maxComp+1)

    range_ = np.linspace(1,maxComp,maxComp)

    axs.bar(range_+0.15,Q2Y,width=0.3,align='center',label='Q2Y', color = "darkblue")
    axs.bar(range_-0.15,R2Y,width=0.3,align='center',label='R2Y', color = "black")
    axs.set_title("R2Y/Q2Y Cell Viability")
    axs.set_xlabel("Number of Components")
    axs.legend(loc=3)


def plotGridSearch(axs, X, Y):
    CVresults_max, CVresults_min, best_params = kmeansPLSR_tuning(X, Y)
    twoC = np.abs(CVresults_min.iloc[:2, 2])
    threeC = np.abs(CVresults_min.iloc[2:5, 2])
    fourC = np.abs(CVresults_min.iloc[5:9, 2])
    fiveC = np.abs(CVresults_min.iloc[9:14, 2])
    sixC = np.abs(CVresults_min.iloc[14:20, 2])

    width = 1
    groupgap = 1

    x1 = np.arange(len(twoC))
    x2 = np.arange(len(threeC))+groupgap+len(twoC)
    x3 = np.arange(len(fourC))+groupgap*2+len(twoC)+len(threeC)
    x4 = np.arange(len(fiveC))+groupgap*3+len(twoC)+len(threeC)+len(fourC)
    x5 = np.arange(len(sixC))+groupgap*4+len(twoC)+len(threeC)+len(fourC)+len(fiveC)

    ind = np.concatenate((x1,x2,x3,x4,x5))

    fig, ax = plt.subplots(figsize=(10,8))
    axs.bar(x1, twoC, width, edgecolor = 'black', color = "g")
    axs.bar(x2, threeC, width, edgecolor = 'black', color = "g")
    axs.bar(x3, fourC, width, edgecolor = 'black', color = "g")
    axs.bar(x4, fiveC, width, edgecolor = "black", color = "g")
    axs.bar(x5, sixC, width, edgecolor = "black", color = "g")

    comps =[]
    for ii in range(2, 7):
        comps.append(list(np.arange(1, ii+1)))
    flattened = [nr for cluster in comps for nr in cluster]

    ax.set_xticks(ind)
    ax.set_xticklabels(flattened, fontsize=10)


def plotActualvsPred(axs, kmeans_plsr, X, Y):
    Y_predictions = np.squeeze(cross_val_predict(kmeans_plsr, X, Y, cv=Y.size))
    axs.scatter(Y, np.squeeze(Y_predictions))
    axs.plot(np.unique(Y), np.poly1d(np.polyfit(Y, np.squeeze(Y_predictions), 1))(np.unique(Y)), color="r")
    axs.set(title="Correlation Measured vs Predicted", xlabel="Actual Y", ylabel="Predicted Y")


def plotScoresLoadings(axs, kmeans_plsr, X, Y, ncl):
    X_scores, Y_scores = kmeans_plsr.fit_transform(X, Y)
    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    PC1_xload, PC2_xload = kmeans_plsr.named_steps.plsr.x_loadings_[:, 0], kmeans_plsr.named_steps.plsr.x_loadings_[:, 1]
    PC1_yload, PC2_yload = kmeans_plsr.named_steps.plsr.y_loadings_[:, 0], kmeans_plsr.named_steps.plsr.y_loadings_[:, 1]

    colors_ = cm.rainbow(np.linspace(0, 1, ncl))

    #Scores
    axs[0].scatter(PC1_scores,PC2_scores)
    for j, txt in enumerate(treatments):
        axs[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    axs[0].set_title('PLSR Model Scores')
    axs[0].set_xlabel('PC1')
    axs[0].set_ylabel('PC2')
    axs[0].axhline(y=0, color='0.25', linestyle='--')
    axs[0].axvline(x=0, color='0.25', linestyle='--')
    axs[0].set_xlim([-6, 6])
    axs[0].set_ylim([-2, 2])

    #Loadings
    numbered=[]
    list(map(lambda v: numbered.append(str(v+1)), range(ncl)))
    for i, txt in enumerate(numbered):
        axs[1].annotate(txt, (PC1_xload[i], PC2_xload[i]))
    axs[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    axs[1].scatter(PC1_yload, PC2_yload, color='#000000', marker='D', label='Cell Viability')
    axs[1].legend(loc=4)
    axs[1].set_title('PLSR Model Loadings (Averaged Clusters)')
    axs[1].set_xlabel('PC1')
    axs[1].set_ylabel('PC2')
    axs[1].axhline(y=0, color='0.25', linestyle='--')
    axs[1].axvline(x=0, color='0.25', linestyle='--')
    axs[1].set_xlim([-1.1, 1.1])
    axs[1].set_ylim([-1.1, 1.1])
