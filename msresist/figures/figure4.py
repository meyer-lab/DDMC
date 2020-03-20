"""
This creates Figure 4.
"""
from .common import subplotLabel, getSetup
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from msresist.sequence_analysis import preprocess_seqs
from msresist.pre_processing import preprocessing, MergeDfbyMean
from msresist.clustering import MassSpecClustering
from msresist.figures.figure2 import plotR2YQ2Y, plotMeasuredVsPredicted, plotScoresLoadings, plotclusteraverages


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 3))

    # blank out first axis for cartoon
    ax[0].axis('off')

    # Cell Viability
    Y_cv1 = pd.read_csv('./msresist/data/Phenotypic_data/GrowthFactors/CV_raw3.csv').iloc[:30, :11]
    Y_cv2 = pd.read_csv('./msresist/data/Phenotypic_data/GrowthFactors/CV_raw4.csv').iloc[:29, :11]

    for ii in range(1, Y_cv2.columns.size):
        Y_cv1.iloc[:, ii] /= Y_cv1.iloc[0, ii]
        Y_cv2.iloc[:, ii] /= Y_cv2.iloc[0, ii]

    Y_cv = MergeDfbyMean(pd.concat([Y_cv1, Y_cv2], axis=0), Y_cv1.columns, "Elapsed")
    Y_cv = Y_cv.reset_index()[Y_cv1.columns]
    v = Y_cv[Y_cv["Elapsed"] == 72].iloc[0, 1:]

    # Phosphorylation data
    X = preprocessing(AXLwt=True, motifs=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    X = preprocess_seqs(X, "Y").sort_values(by="Protein")

    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    treatments = list(d.index)

    ncl = 3
    distance_method = "PAM250"
    GMMweight = 0.75
    b = ncl + 1

    MSC = MassSpecClustering(i, ncl, GMMweight=GMMweight, distance_method=distance_method).fit(d, v)
    centers = MSC.transform(d)

    ncomp = 2
    mixedCl_plsr = Pipeline([('mixedCl', MassSpecClustering(i, ncl, GMMweight=GMMweight, distance_method=distance_method)), ('plsr', PLSRegression(ncomp))])

    fit = mixedCl_plsr.fit(d, v)
    centers = mixedCl_plsr.named_steps.mixedCl.transform(d)
    
    plotR2YQ2Y(ax[1], mixedCl_plsr, d, v, cv=2, b=b)
    
    plotMeasuredVsPredicted(ax[2], mixedCl_plsr, d, v)

    plotScoresLoadings(ax[3:5], fit, centers, v, ncl, treatments, CV=2)

    plotclusteraverages(ax[5], centers.T, treatments)

    # Add subplot labels
    subplotLabel(ax)

    return f
