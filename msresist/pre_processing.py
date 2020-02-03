""" This scripts handles all the pre-processing required to merge and transform the raw mass spec biological replicates. """

import os
import numpy as np
import pandas as pd
from scipy import stats
from .sequence_analysis import FormatName, pYmotifs


path = os.path.dirname(os.path.abspath(__file__))


###-------------------------- Pre-processing Raw Data --------------------------###


def preprocessing(AXLwt=False, Axlmuts_Erl=False, Axlmuts_ErlF154=False, C_r=False, motifs=False, Vfilter=False, FCfilter=False, log2T=False, rawdata=False, mc_row=False, mc_col=False):
    """ Input: Raw MS bio-replicates. Output: Mean-centered merged data set.
    1. Concatenation, 2. log-2 transformation, 3. Mean-Center, 4. Merging, 5. Fold-change,
    6. Filters: 'Vfilter' filters by correlation when 2 overlapping peptides or std cutoff if >= 3.
    Note 1: 'motifs' redefines peptide sequences as XXXXXyXXXXX which affects merging.
    Note 2: Data is converted back to linear scale before filtering so 'log2T=True' to use log-scale for analysis."""
    filesin = list()

    if AXLwt:
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/20180817_JG_AM_TMT10plex_R1_psms_raw.csv"), header=0))
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/20190214_JG_AM_PC9_AXL_TMT10_AC28_R2_PSMs_raw.csv"), header=0))
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/CombinedBR3_TR1&2_raw.csv"), header=0))
    if Axlmuts_Erl:
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/PC9_mutants_unstim_BR1_raw.csv"), header=0))
    if Axlmuts_ErlF154:
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/PC9_mutants_ActivatingAb_BR1_raw.csv"), header=0))

    ABC = MeanCenter(Log2T(pd.concat(filesin)), mc_row, mc_col)

    longnames, shortnames = FormatName(ABC)
    ABC["Protein"] = longnames
    ABC = ABC.assign(Abbv=shortnames)
    merging_indices = list(ABC.columns[:3]) + ["Abbv"]

    if rawdata:
        return ABC

    if motifs:
        ABC = pYmotifs(ABC, longnames)
        merging_indices += ["Position"]

    if Vfilter:
        ABC = VFilter(ABC, merging_indices)

    ABC = MergeDfbyMean(
        ABC.copy(),
        ABC.iloc[3:13],
        merging_indices
    ).reset_index()[merging_indices[:3] + list(ABC.columns[3:13]) + merging_indices[3:]]

    if FCfilter:
        ABC = FoldChangeFilter(ABC)

    if not log2T:
        ABC = LinearFoldChange(ABC)

    return ABC[merging_indices + list(filesin[0].columns[3:13])]


def MergeDfbyMean(X, values, indices):
    """ Compute mean across duplicates. """
    return pd.pivot_table(X, values=values, index=indices, aggfunc=np.mean)


def LinearFoldChange(X):
    """ Convert to linear fold-change from log2 mean-centered. """
    X.iloc[:, 3:13] = pd.DataFrame(np.power(2, X.iloc[:, 3:13])).div(np.power(2, X.iloc[:, 3]), axis=0)
    return X


def FoldChangeToControl(X):
    """ Convert to fold-change to control. """
    X.iloc[:, 3:13] = X.iloc[:, 3:13].div(X.iloc[:, 3], axis=0)
    return X


def Log2T(X):
    """ Convert to log2 scale keeping original sign. """
    X.iloc[:, 3:13] = np.log2(X.iloc[:, 3:13])
    return X


def MeanCenter(X, mc_row, mc_col):
    """ Mean centers each row of values. logT also optionally log2-transforms. """
    if mc_row:
        X.iloc[:, 3:13] = X.iloc[:, 3:13].sub(X.iloc[:, 3:13].mean(axis=1), axis=0)
    if mc_col:
        X.iloc[:, 3:13] = X.iloc[:, 3:13].sub(X.iloc[:, 3:13].mean(axis=0), axis=1)
    return X


def VarianceFilter(X, varCut=0.1):
    """ Filter rows for those containing more than cutoff variance. Variance across conditions per peptide.
    Note this should only be used with log-scaled, mean-centered data. """
    Xidx = np.var(X.iloc[:, 3:13].values, axis=1) > varCut
    return X.iloc[Xidx, :]  # .iloc keeps only those peptide labeled as "True"


def FoldChangeFilter(X):
    """ Filter rows for those containing more than a two-fold change.
    Note this should only be used with linear-scale data normalized to the control. """
    XX = LinearFoldChange(X.copy())
    Xidx = np.any(XX.iloc[:, 3:13].values <= 0.5, axis=1) | np.any(XX.iloc[:, 3:13].values >= 2.0, axis=1)
    return X.iloc[Xidx, :]


###------------ Filter by variance (stdev or range/pearson's) ------------------###


def VFilter(ABC, merging_indices):
    """ Filter based on variability across recurrent peptides in MS biological replicates """
    NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

    NonRecTable = BuildMatrix(NonRecPeptides, ABC)
    NonRecTable = NonRecTable.assign(BioReps=list("1" * NonRecTable.shape[0]))
    NonRecTable = NonRecTable.assign(r2_Std=list(["N/A"] * NonRecTable.shape[0]))

    CorrCoefPeptides = BuildMatrix(CorrCoefPeptides, ABC)
    DupsTable = CorrCoefFilter(CorrCoefPeptides)
    DupsTable = MergeDfbyMean(DupsTable, DupsTable.iloc[3:13], merging_indices)
    DupsTable = DupsTable.assign(BioReps=list("2" * DupsTable.shape[0])).reset_index()

    StdPeptides = BuildMatrix(StdPeptides, ABC)
    TripsTable = TripsMeanAndStd(StdPeptides, merging_indices + ["BioReps"])
    TripsTable = FilterByStdev(TripsTable)

    merging_indices += ["BioReps", "r2_Std"]

    ABC = pd.concat([
        NonRecTable,
        DupsTable,
        TripsTable
    ]
    ).reset_index()[merging_indices[:3] + list(ABC.columns[3:13]) + merging_indices[3:]]

    # Including non-overlapping peptides
    return ABC


def MapOverlappingPeptides(ABC):
    """ Find recurrent peptides across biological replicates. Grouping those showing up 2 to later calculate
    correlation, those showing up >= 3 to take the std. Those showing up 1 can be included or not in the final data set.
    Final dfs are formed by 'Name', 'Peptide', '#Recurrences'. """
    dups = pd.pivot_table(ABC, index=["Protein", "Sequence"], aggfunc="size").sort_values()
    dups = pd.DataFrame(dups).reset_index()
    dups.columns = [ABC.columns[0], ABC.columns[1], "Recs"]
    NonRecPeptides = dups[dups["Recs"] == 1]
    RangePeptides = dups[dups["Recs"] == 2]
    StdPeptides = dups[dups["Recs"] >= 3]
    return NonRecPeptides, RangePeptides, StdPeptides


def BuildMatrix(peptides, ABC):
    """ Map identified recurrent peptides in the concatenated data set to generate complete matrices with values.
    If recurrent peptides = 2, the correlation coefficient is included in a new column. """
    ABC = ABC.reset_index().set_index(["Sequence", "Protein"], drop=False)

    corrcoefs, peptideslist, bioReps = [], [], []
    for idx, seq in enumerate(peptides.iloc[:, 1]):
        name = peptides.iloc[idx, 0]

        # Skip blank
        if name == "(blank)":
            continue

        pepts = ABC.loc[seq, name]
        pepts = pepts.iloc[:, 1:]
        names = pepts.iloc[:, 0]

        if len(pepts) == 1:
            peptideslist.append(pepts.iloc[0, :])
        elif len(pepts) == 2 and len(set(names)) == 1:
            fc = LinearFoldChange(pepts.iloc[:, 3:13].copy())
            corrcoef, _ = stats.pearsonr(fc.iloc[0, 3:13], fc.iloc[1, 3:13])
            for i in range(len(pepts)):
                corrcoefs.append(np.round(corrcoef, decimals=2))
                peptideslist.append(pepts.iloc[i, :])
        elif len(pepts) >= 3 and len(set(names)) == 1:
            for i in range(len(pepts)):
                peptideslist.append(pepts.iloc[i, :])
                bioReps.append(len(pepts))
        else:
            print("check this", pepts)

    if corrcoefs:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True).assign(r2_Std=corrcoefs)

    elif bioReps:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True).assign(BioReps=bioReps)

    else:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True)

    return matrix


def CorrCoefFilter(X, corrCut=0.6):
    """ Filter rows for those containing more than a correlation threshold. """
    Xidx = X.iloc[:, -1].values >= corrCut
    return X.iloc[Xidx, :]


def DupsMeanAndRange(duplicates, header):
    """ Merge all duplicates by mean and range across conditions. Note this builds a multilevel header
    meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Range). """
    func_dup = {}
    for i in header[3:13]:
        func_dup[i] = np.mean, np.ptp
    ABC_dups_avg = pd.pivot_table(duplicates, values=header[3:13], index=header[:2], aggfunc=func_dup)
    ABC_dups_avg = ABC_dups_avg.reset_index()[header]
    return ABC_dups_avg


def TripsMeanAndStd(triplicates, merging_indices):
    """ Merge all triplicates by mean and standard deviation across conditions. Note this builds a multilevel header
    meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Std). """
    func_tri = {}
    for i in triplicates.columns[3:13]:
        func_tri[i] = np.mean, np.std
    X = pd.pivot_table(triplicates, values=triplicates.columns[3:13], index=merging_indices, aggfunc=func_tri)
    return X.reset_index()


def FilterByRange(X, rangeCut=0.4):
    """ Filter rows for those containing more than a range threshold. """
    Rg = X.iloc[:, X.columns.get_level_values(1) == "ptp"]
    Xidx = np.all(Rg.values <= rangeCut, axis=1)
    return X.iloc[Xidx, :]


def FilterByStdev(X, stdCut=0.4):
    """ Filter rows for those containing more than a standard deviation threshold. """
    Stds = X.iloc[:, X.columns.get_level_values(1) == "std"]
    StdMeans = list(np.round(Stds.mean(axis=1), decimals=2))
#     display(pd.DataFrame(StdMeans))
    Xidx = np.all(Stds.values <= stdCut, axis=1)
#     display(Stds)
#     display(pd.DataFrame(Xidx))
    if "Position" in X.columns:
        Means = pd.concat([X.iloc[:, :6], X.iloc[:, X.columns.get_level_values(1) == "mean"]], axis=1)
    else:
        Means = pd.concat([X.iloc[:, :5], X.iloc[:, X.columns.get_level_values(1) == "mean"]], axis=1)
    Means = Means.iloc[Xidx, :]
    Means.columns = Means.columns.droplevel(1)
    StdMeans = pd.DataFrame(StdMeans).iloc[Xidx, :]
    return Means.assign(r2_Std=StdMeans)


def peptidefinder(X, loc, Protein=False, Abbv=False, Sequence=False):
    """ Search for a specific peptide either by name or sequence. """
    if Protein:
        found = X[X["Protein"].str.contains(loc)]
    if Abbv:
        found = X[X["Abbv"].str.contains(loc)]
    if Sequence:
        found = X[X["Sequence"].str.contains(loc)]
    return found
