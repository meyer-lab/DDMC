"""
All functions relaed to GSEA analysis of clusters
"""

import pickle
import pandas as pd
import mygene
from msresist.pre_processing import preprocessing, filter_NaNpeptides


path = "/Users/creixell/Desktop/"


def translate_geneIDs(X, labels, toID="entrezgene", export=False, outpath="GSEA_Input.csv"):
    """ Generate GSEA clusterProfiler input data. Translate gene accessions.
    In this case to ENTREZID by default. """
    X["Clusters"] = labels
    X.index = list(range(X.shape[0]))
    mg = mygene.MyGeneInfo()
    gg = mg.querymany(list(X["Gene"]), scopes="symbol", fields="entrezgene", species="human", returnall=False, as_dataframe=True)
    aa = dict(zip(list(gg.index), list(gg["entrezgene"])))
    for ii in range(X.shape[0]):
        X.loc[ii, "Gene"] = aa[X.loc[ii, "Gene"]]
    if export:
        X[["Gene", "Clusters"]].to_csv(outpath)
    return X[["Gene", "Clusters"]]
