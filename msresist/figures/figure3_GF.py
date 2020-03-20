# """
# This creates Figure 3.
# """
# import os
# import pandas as pd
# import numpy as np
# import scipy as sp
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from .common import subplotLabel, getSetup
# from sklearn.model_selection import cross_val_predict
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from msresist.parameter_tuning import MSclusPLSR_tuning, kmeansPLSR_tuning
# from msresist.plsr import Q2Y_across_components, R2Y_across_components
# from msresist.clustering import MassSpecClustering
# from msresist.sequence_analysis import preprocess_seqs
# import matplotlib.colors as colors
# import matplotlib.cm as cm
# import seaborn as sns
# from ..sequence_analysis import FormatName, pYmotifs
# from ..pre_processing import preprocessing, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd, MergeDfbyMean
# from ..figures.figure1 import plotpca_ScoresLoadings, plotVarReplicates, plotProteinSites

# path = os.path.dirname(os.path.abspath(__file__))


# def makeFigure():
#     """Get a list of the axis objects and create a figure"""
#     # Get list of axis objects
#     ax, f = getSetup((10, 16), (5, 3))

#     # blank out first axis for cartoon
#     # ax[0].axis('off')

#     # Read in data
#     Y_cv1 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/GrowthFactors/CV_raw3.csv')).iloc[:30, :11]
#     Y_cv2 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/GrowthFactors/CV_raw4.csv')).iloc[:29, :11]

#     # Assert that there's no significant influence of the initial seeding density
#     t = 72
#     cv1t0 = Y_cv1[Y_cv1["Elapsed"] == 0].iloc[0, 1:]
#     cv1 = Y_cv1[Y_cv1["Elapsed"] == t].iloc[0, 1:] / cv1t0

#     cv2t0 = Y_cv2[Y_cv2["Elapsed"] == 0].iloc[0, 1:]
#     cv2 = Y_cv2[Y_cv2["Elapsed"] == t].iloc[0, 1:] / cv2t0

#     assert sp.stats.pearsonr(cv1t0, cv1)[1] > 0.05
#     assert sp.stats.pearsonr(cv2t0, cv2)[1] > 0.05

#     # Phosphorylation data
#     X = preprocessing(AXLwt=True, motifs=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
#     X = preprocess_seqs(X, "Y").sort_values(by="Protein")

#     d = X.select_dtypes(include=['float64']).T
#     i = X.select_dtypes(include=['object'])

#     treatments = list(d.index)

#     # A: Cell Viability Endpoint
#     BarPlot_FCendpoint(ax[0], cv1, cv2, t, list(Y_cv1.columns[1:]))

#     # B: blank out second axis for signaling ClusterMap
#     ax[1].axis('off')

#     # C&D: Scores and Loadings MS data
#     plotpca_ScoresLoadings(ax[2:4], d)

#     # E: Variability across overlapping peptides in MS replicates
#     X = preprocessing(AXLwt=True, rawdata=True)
#     plotVarReplicates(ax[4:6], X)

#     # F: AXL
#     df = preprocessing(AXLwt=True, motifs=True, Vfilter=False, FCfilter=False, log2T=False, FCtoUT=True, mc_row=True)
#     plotProteinSites(ax[6], df.copy(), "AXL", "AXL")
#     plotProteinSites(ax[7], df.copy(), "EGFR", "EGFR")
#     plotProteinSites(ax[8], df.copy(), "MET", "MET")
#     plotProteinSites(ax[9], df.copy(), "ERBB2", "HER2")
#     plotProteinSites(ax[10], df.copy(), "ERBB3", "HER3")
#     plotProteinSites(ax[11], df.copy(), "GAB1", "GAB1")
#     plotProteinSites(ax[12], df.copy(), "MAPK3", "ERK1")
#     plotProteinSites(ax[13], df.copy(), "MAPK1", "ERK3")
#     plotProteinSites(ax[14], df.copy(), "YES1", "YES1")

#     # Add subplot labels
#     subplotLabel(ax)

#     return f


# def BarPlot_FCendpoint(ax, x, y, t, lines):
#     c = pd.concat([x, y]).reset_index()
#     c.columns = ["lines", "Cell Viability (fold-change t=0)"]

#     ax = sns.barplot(x="lines",
#                      y="Cell Viability (fold-change t=0)",
#                      data=c, ci="sd",
#                      color="darkcyan")

#     ax.set_title("t =" + str(t) + "h")
