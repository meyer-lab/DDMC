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
from msresist.plsr import Q2Y_across_components, R2Y_across_components, plotMeasuredVsPredicted
from msresist.clustering import MyOwnKMEANS
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from msresist.pre_processing import preprocessing, MergeDfbyMean
import warnings
warnings.simplefilter("ignore")


path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 10), (2, 3))

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
    ABC_mc = preprocessing(motifs=True, Vfilter=True, FCfilter=True, log2T=True)

    header = ABC_mc.columns
    treatments = ABC_mc.columns[2:]

    data = ABC_mc.iloc[:,2:].T

    #Set up model pipeline
    ncl, ncomp = 4, 2
    estimators = [('kmeans', MyOwnKMEANS(ncl)), ('plsr', PLSRegression(ncomp))]
    kmeans_plsr = Pipeline(estimators)
    
    colors_ = cm.rainbow(np.linspace(0, 1, ncl))

    plotR2YQ2Y(ax[0], ncl, data, Y_cv)

    plotGridSearch(ax[1], data ,Y_cv)

    plotMeasuredVsPredicted(ax[2], kmeans_plsr, data, Y_cv)

    plotScoresLoadings(ax[3:5], kmeans_plsr, data, Y_cv, ncl, treatments, colors_)
    
    clusteraverages(ax[5], data, kmeans_plsr, colors_, treatments)

    # Add subplot labels
    subplotLabel(ax)

    return f

def plotR2YQ2Y(ax, ncl, X, Y):
    kmeans = MyOwnKMEANS(ncl).fit(X, Y)
    centers = kmeans.transform(X)

    maxComp = ncl
    Q2Y = Q2Y_across_components(centers, Y, maxComp+1)
    R2Y = R2Y_across_components(centers, Y, maxComp+1)

    range_ = np.linspace(1,maxComp,maxComp)

    ax.bar(range_+0.15,Q2Y,width=0.3,align='center',label='Q2Y', color = "darkblue")
    ax.bar(range_-0.15,R2Y,width=0.3,align='center',label='R2Y', color = "black")
    ax.set_title("R2Y/Q2Y Cell Viability")
    ax.set_xlabel("Number of Components")
    ax.legend(loc=3)


def plotGridSearch(ax, X, Y):
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

    ax.bar(x1, twoC, width, edgecolor = 'black', color = "g")
    ax.bar(x2, threeC, width, edgecolor = 'black', color = "g")
    ax.bar(x3, fourC, width, edgecolor = 'black', color = "g")
    ax.bar(x4, fiveC, width, edgecolor = "black", color = "g")
    ax.bar(x5, sixC, width, edgecolor = "black", color = "g")

    comps =[]
    for ii in range(2, 7):
        comps.append(list(np.arange(1, ii+1)))
    flattened = [nr for cluster in comps for nr in cluster]

    ax.set_xticks(ind)
    ax.set_xticklabels(flattened, fontsize=10)
    ax.set_xlabel("Number of Components per Cluster")
    ax.set_ylabel("Mean-Squared Error (MSE)")


def plotScoresLoadings(ax, kmeans_plsr, X, Y, ncl, treatments, colors_):
    X_scores, Y_scores = kmeans_plsr.fit_transform(X, Y)
    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    PC1_xload, PC2_xload = kmeans_plsr.named_steps.plsr.x_loadings_[:, 0], kmeans_plsr.named_steps.plsr.x_loadings_[:, 1]
    PC1_yload, PC2_yload = kmeans_plsr.named_steps.plsr.y_loadings_[:, 0], kmeans_plsr.named_steps.plsr.y_loadings_[:, 1]

    #Scores
    ax[0].scatter(PC1_scores,PC2_scores)
    for j, txt in enumerate(treatments):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    ax[0].set_title('PLSR Model Scores')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    ax[0].axhline(y=0, color='0.25', linestyle='--')
    ax[0].axvline(x=0, color='0.25', linestyle='--')
    ax[0].set_xlim([-6, 6])
    ax[0].set_ylim([-2, 2])

    #Loadings
    numbered=[]
    list(map(lambda v: numbered.append(str(v+1)), range(ncl)))
    for i, txt in enumerate(numbered):
        ax[1].annotate(txt, (PC1_xload[i], PC2_xload[i]))
    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].scatter(PC1_yload, PC2_yload, color='#000000', marker='D', label='Cell Viability')
    ax[1].legend(loc=4)
    ax[1].set_title('PLSR Model Loadings (Averaged Clusters)')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].axhline(y=0, color='0.25', linestyle='--')
    ax[1].axvline(x=0, color='0.25', linestyle='--')
    ax[1].set_xlim([-1.1, 1.1])
    ax[1].set_ylim([-1.1, 1.1])
    
def clusteraverages(ax, X, model_plsr, colors_, treatments):
    centers = model_plsr.named_steps.kmeans.transform(X).T
    for i in range(centers.shape[0]):
        ax.plot(centers.iloc[i,:], label = "cluster "+str(i+1), color = colors_[i])
    ax.legend()
    
    ax.set_xticks(np.arange(centers.shape[1]))
    ax.set_xticklabels(treatments, rotation=70, rotation_mode="anchor")
    ax.set_ylabel("normalized signal")
