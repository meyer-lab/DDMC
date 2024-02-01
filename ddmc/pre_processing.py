""" This scripts handles all the pre-processing required to merge and transform the raw mass spec biological replicates. """

import os
import numpy as np
import pandas as pd
from collections import defaultdict


path = os.path.dirname(os.path.abspath(__file__))


###-------------------------- Pre-processing MS data --------------------------###
def preprocessCPTAC():
    """Replace patient identifiers, fill NaNs, and make it compatible with current code."""
    X = pd.read_csv(
        os.path.join(
            path,
            "./data/MS/CPTAC/CPTAC3_Lung_Adeno_Carcinoma_Phosphoproteome.phosphopeptide.tmt10.csv",
        )
    )
    d = X.iloc[:, 1:-3]
    X = pd.concat(
        [X.iloc[:, 0], X.iloc[:, -3:], d.loc[:, d.columns.str.contains("CPT")]], axis=1
    )
    X = filter_NaNpeptides(X, cut=0.2)

    n = pd.read_csv(
        os.path.join(
            path,
            "./data/MS/CPTAC/S046_BI_CPTAC3_LUAD_Discovery_Cohort_Samples_r1_May2019.csv",
        )
    )
    bi_id = list(n[~n["Broad Sample.ID"].str.contains("IR")].iloc[:, 1])
    X.columns = ["Sequence"] + list(X.columns[1:4]) + bi_id

    return X.drop("Organism", axis=1)


def filter_NaNpeptides(X, cut=False, tmt=False):
    """Filter peptides that have a given minimum percentage of completeness or number of TMT experiments."""
    d = X.select_dtypes(include=["float64"])
    if cut:
        Xidx = np.count_nonzero(~np.isnan(d), axis=1) / d.shape[1] >= cut
    else:
        idx_values = FindIdxValues(X)
        dict_ = defaultdict(list)
        for i in range(idx_values.shape[0]):
            dict_[idx_values[i, 0]].append(idx_values[i, -1])
        Xidx = [len(set(dict_[i])) >= tmt for i in range(X.shape[0])]
    return X.iloc[Xidx, :]

def FindIdxValues(X):
    """Find the patient indices corresponding to all non-missing values grouped in TMT experiments."""
    data = X.select_dtypes(include=["float64"])
    idx = np.argwhere(~np.isnan(data.values))
    idx[:, 1] += 4  # add ID variable columns
    StoE = pd.read_csv("ddmc/data/MS/CPTAC/IDtoExperiment.csv")
    assert all(StoE.iloc[:, 0] == data.columns), "Sample labels don't match."
    StoE = StoE.iloc[:, 1].values
    tmt = [[StoE[idx[ii][1] - 4]] for ii in range(idx.shape[0])]
    return np.append(idx, tmt, axis=1)


def MergeDfbyMean(X, values, indices):
    """Compute mean across duplicates."""
    return pd.pivot_table(X, values=values, index=indices, aggfunc="mean")


def LinearFoldChange(X, data_headers, FCto):
    """Convert to linear fold-change from log2 mean-centered."""
    X[data_headers] = pd.DataFrame(np.power(2, X[data_headers])).div(
        np.power(2, X[FCto]), axis=0
    )
    return X


def Linear(X, data_headers):
    """Convert to linear from log2 mean-centered."""
    X[data_headers] = pd.DataFrame(np.power(2, X[data_headers]))
    return X


def FoldChangeToControl(X, data_headers):
    """Convert to fold-change to control."""
    X[data_headers] = X[data_headers].div(X.iloc[:, 3], axis=0)
    return X


def Log2T(X):
    """Convert to log2 scale keeping original sign."""
    data_headers = X.select_dtypes(include=["float64"]).columns
    X[data_headers] = np.log2(X[data_headers])
    return X


def VarianceFilter(X, data_headers, varCut=0.1):
    """Filter rows for those containing more than cutoff variance. Variance across conditions per peptide.
    Note this should only be used with log-scaled, mean-centered data."""
    Xidx = np.var(X[data_headers].values, axis=1) > varCut
    return X.iloc[Xidx, :]


def FoldChangeFilterToControl(X, data_headers, FCto, cutoff=0.4):
    """Filter rows for those containing more than a two-fold change.
    Note this should only be used with linear-scale data normalized to the control."""
    XX = LinearFoldChange(X.copy(), data_headers, FCto)
    Xidx = np.any(XX[data_headers].values <= 1 - cutoff, axis=1) | np.any(
        XX[data_headers].values >= 1 + cutoff, axis=1
    )
    return X.iloc[Xidx, :]


def FoldChangeFilterBasedOnMaxFC(X, data_headers, cutoff=0.5):
    """Filter rows for those containing an cutoff% change of the maximum vs minimum fold-change
    across every condition."""
    XX = Linear(X.copy(), data_headers)
    X_ToMin = XX[data_headers] / XX[data_headers].min(axis=0)
    Xidx = np.any(X_ToMin.values >= X_ToMin.max().values * cutoff, axis=1)
    return X.iloc[Xidx, :]


def separate_sequence_and_abundance(ms_df: pd.DataFrame):
    # by default, we assume that ms_df is composed of "Gene", "Sequence",
    # "Position", and sample columns
    sample_cols = [
        col
        for col in ms_df.columns
        if col not in ("Gene", "Sequence", "Position", "Protein")
    ]
    return ms_df["Sequence"].copy(), ms_df[sample_cols].copy()


