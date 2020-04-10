# """
# This creates Figure 4.
# """

# from .common import subplotLabel, getSetup
# from msresist.pre_processing import preprocessing
# import pandas as pd
# import numpy as np
# from msresist.figures.figure3 import plotclusteraverages
# from msresist.clustering import MassSpecClustering


# def makeFigure():
#     """Get a list of the axis objects and create a figure"""
#     # Get list of axis objects
#     ax, f = getSetup((10, 10), (1, 1))
#     X = preprocessing(CPTAC=True, log2T=True)

#     d = X.select_dtypes(include=['float64']).T
#     i = X.select_dtypes(include=['object'])
    
#     dred = d.iloc[:, :4000]
#     ired = i.iloc[:4000, :]
    
#     distance_method = "Binomial"
#     ncl = 3
#     GMMweight = 0.5

#     MSC = MassSpecClustering(ired, ncl, GMMweight=GMMweight, distance_method=distance_method, n_runs=1).fit(dred, "NA")
    
#     plotclusteraverages(ax[0], MSC.transform(dred).T, dred.index)

#     # Add subplot labels
#     subplotLabel(ax)

#     return f