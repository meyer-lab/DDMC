"""
All functions relaed to GSEA analysis of clusters
"""

import pickle
import pandas as pd
import mygene
from msresist.pre_processing import preprocessing, filter_NaNpeptides


path = "/Users/creixell/Desktop/"
def translate_geneIDs(d, geneID, path, col, toID="entrezgene", export=False):
    """ Generate GSEA clusterProfiler input data. Translate gene accessions. 
    In this case to ENTREZID by default. """
    if type == "CPTAC":
        with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
            model = pickle.load(p)[0]
        X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
        X = filter_NaNpeptides(X, tmt=2)
    elif type == "AXL":
        with open("msresist/data/pickled_models/AXLmodel_PAM250_W2-5_5CL", "rb") as m:
            model = pickle.load(m)
            X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    X["Clusters"] = model.labels()
    mg = mygene.MyGeneInfo()
    gg = mg.querymany(list(X["Gene"]), scopes="symbol", fields="entrezgene", species="human", returnall=False, as_dataframe=True)
    aa = dict(zip(list(gg.index), list(gg["entrezgene"])))
    for ii in range(X.shape[0]):
        X.iloc[ii, 2] = aa[X.iloc[ii, 2]]
    if export:
        X[["Gene", "Clusters"]].to_csv("CPTAC_GSEA_Input.csv")
    return X[["Gene", "Clusters"]]