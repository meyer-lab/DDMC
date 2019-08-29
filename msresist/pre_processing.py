""" This scripts handles all the pre-processing required to merge and transform the raw mass spec biological replicates. """
import os
import numpy as np
import pandas as pd
from scipy import stats
from .sequence_analysis import FormatName, pYmotifs


path = os.path.dirname(os.path.abspath(__file__))


###-------------------------- Pre-processing Raw Data --------------------------###


def preprocessing(A_r=True, B_r=True, C_r=True, motifs=False, Vfilter=False, FCfilter=False, log2T=False, rawdata=False):
    """ Input: Raw MS bio-replicates. Output: Mean-centered merged data set.
    1. Concatenation, 2. log-2 transformation, 3. Mean-Center, 4. Merging, 5. Fold-change,
    6. Filters: 'Vfilter' filters by correlation when 2 overlapping peptides or std cutoff if >= 3.
    Note 1: 'motifs' redefines peptide sequences as XXXXXyXXXXX which affects merging.
    Note 2: Data is converted back to linear scale before filtering so 'log2T=True' to use log-scale for analysis."""
    filesin = list()

    if A_r:
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/20180817_JG_AM_TMT10plex_R1_psms_raw.csv"), header=0))
    if B_r:
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/20190214_JG_AM_PC9_AXL_TMT10_AC28_R2_PSMs_raw.csv"), header=0))
    if C_r:
        filesin.append(pd.read_csv(os.path.join(path, "./data/Raw/CombinedBR3_TR1&2_raw.csv"), header=0))

    ABC_conc_mc = MeanCenter(Log2T(pd.concat(filesin)))

    ABC_names = FormatName(ABC_conc_mc)
    ABC_conc_mc["Master Protein Descriptions"] = ABC_names

    if rawdata:
        return ABC_conc_mc

    if motifs:
        ABC_conc_mc = pYmotifs(ABC_conc_mc, ABC_names)

    if Vfilter:
        ABC_conc_mc = VFilter(ABC_conc_mc)

    ABC_conc_mc = MergeDfbyMean(ABC_conc_mc.copy(), filesin[0].columns[2:], ["Master Protein Descriptions", "peptide-phosphosite"])
    ABC_conc_mc = ABC_conc_mc.reset_index()[filesin[0].columns]

    if FCfilter:
        ABC_conc_mc = FoldChangeFilter(ABC_conc_mc)

    if not log2T:
        ABC_conc_mc = LinearFoldChange(ABC_conc_mc)

    return ABC_conc_mc


def MergeDfbyMean(X, values, index):
    """ Compute mean across duplicates. """
    return pd.pivot_table(X, values=values, index=index, aggfunc=np.mean)


def LinearFoldChange(X):
    """ Convert to linear fold-change from log2 mean-centered. """
    X.iloc[:, 2:] = np.power(2, X.iloc[:, 2:]).div(np.power(2, X.iloc[:, 2]), axis=0)
    return X


def FoldChangeToControl(X):
    """ Convert to fold-change to control. """
    X.iloc[:, 2:] = X.iloc[:, 2:].div(X.iloc[:, 2], axis=0)
    return X


def Log2T(X):
    """ Convert to log2 scale keeping original sign. """
    X.iloc[:, 2:] = np.log2(X.iloc[:, 2:])
    return X


def MeanCenter(X):
    """ Mean centers each row of values. logT also optionally log2-transforms. """
    X.iloc[:, 2:] = X.iloc[:, 2:].sub(X.iloc[:, 2:].mean(axis=1), axis=0)
    return X


def VarianceFilter(X, varCut=0.1):
    """ Filter rows for those containing more than cutoff variance. Variance across conditions per peptide.
    Note this should only be used with log-scaled, mean-centered data. """
    Xidx = np.var(X.iloc[:, 2:].values, axis=1) > varCut
    return X.iloc[Xidx, :]  # .iloc keeps only those peptide labeled as "True"


def FoldChangeFilter(X):
    """ Filter rows for those containing more than a two-fold change.
    Note this should only be used with linear-scale data normalized to the control. """
    XX = LinearFoldChange(X.copy())
    Xidx = np.any(XX.iloc[:, 2:].values <= 0.5, axis=1) | np.any(XX.iloc[:, 2:].values >= 2.0, axis=1)
    return X.iloc[Xidx, :]


###------------ Filter by variance (stdev or range/pearson's) ------------------###


def VFilter(ABC_conc_mc):
    NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC_conc_mc)

    NonRecTable = BuildMatrix(NonRecPeptides, ABC_conc_mc)

    CorrCoefPeptides = BuildMatrix(CorrCoefPeptides, ABC_conc_mc)
    DupsTable = CorrCoefFilter(CorrCoefPeptides)
    DupsTable = MergeDfbyMean(CorrCoefPeptides, DupsTable.columns[2:], ["Master Protein Descriptions", "peptide-phosphosite"])
    DupsTable = DupsTable.reset_index()[ABC_conc_mc.columns]

    StdPeptides = BuildMatrix(StdPeptides, ABC_conc_mc)
    TripsTable = TripsMeanAndStd(StdPeptides, ABC_conc_mc.columns)
    TripsTable = FilterByStdev(TripsTable)
    TripsTable.columns = ABC_conc_mc.columns

    #     ABC_mc = pd.concat([DupsTable, TripsTable])   # Leaving non-overlapping peptides out
    ABC_mc = pd.concat([NonRecTable, DupsTable, TripsTable])  # Including non-overlapping peptides
    return ABC_mc


def MapOverlappingPeptides(ABC):
    """ Find recurrent peptides across biological replicates. Grouping those showing up 2 to later calculate
    correlation, those showing up >= 3 to take the std. Those showing up 1 can be included or not in the final data set.
    Final dfs are formed by 'Name', 'Peptide', '#Recurrences'. """
    dups = pd.pivot_table(ABC, index=["Master Protein Descriptions", "peptide-phosphosite"], aggfunc="size").sort_values()
    #     dups_counter = {i: list(dups).count(i) for i in list(dups)}
    dups = pd.DataFrame(dups).reset_index()
    dups.columns = [ABC.columns[1], ABC.columns[0], "Recs"]
    NonRecPeptides = dups[dups["Recs"] == 1]
    RangePeptides = dups[dups["Recs"] == 2]
    StdPeptides = dups[dups["Recs"] >= 3]
    return NonRecPeptides, RangePeptides, StdPeptides


def BuildMatrix(peptides, ABC):
    """ Map identified recurrent peptides in the concatenated data set to generate complete matrices with values.
    If recurrent peptides = 2, the correlation coefficient is included in a new column. """
    peptideslist = []
    corrcoefs = []
    for idx, seq in enumerate(peptides.iloc[:, 1]):
        name = peptides.iloc[idx, 0]
        pepts = ABC.reset_index().set_index(["peptide-phosphosite", "Master Protein Descriptions"], drop=False).loc[seq, name]
        pepts = pepts.iloc[:, 1:]
        names = pepts.iloc[:, 1]
        if name == "(blank)":
            continue
        elif len(pepts) == 1:
            peptideslist.append(pepts.iloc[0, :])
        elif len(pepts) == 2 and len(set(names)) == 1:
            corrcoef, _ = stats.pearsonr(pepts.iloc[0, 2:], pepts.iloc[1, 2:])
            for i in range(len(pepts)):
                corrcoefs.append(corrcoef)
                peptideslist.append(pepts.iloc[i, :])
        elif len(pepts) >= 3 and len(set(names)) == 1:
            for i in range(len(pepts)):
                peptideslist.append(pepts.iloc[i, :])
        else:
            print("check this", pepts)

    if corrcoefs:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True)
        matrix = matrix.assign(CorrCoefs=corrcoefs)

    else:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True)

    return matrix


def CorrCoefFilter(X, corrCut=0.5):
    """ Filter rows for those containing more than a correlation threshold. """
    XX = LinearFoldChange(X.copy())
    Xidx = XX.iloc[:, 12].values >= corrCut
    return X.iloc[Xidx, :]


def DupsMeanAndRange(duplicates, header):
    """ Merge all duplicates by mean and range across conditions. Note this builds a multilevel header
    meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Range). """
    func_dup = {}
    for i in header[2:]:
        func_dup[i] = np.mean, np.ptp
    ABC_dups_avg = pd.pivot_table(duplicates, values=header[2:], index=["Master Protein Descriptions", "peptide-phosphosite"], aggfunc=func_dup)
    ABC_dups_avg = ABC_dups_avg.reset_index()[header]
    return ABC_dups_avg


def TripsMeanAndStd(triplicates, header):
    """ Merge all triplicates by mean and standard deviation across conditions. Note this builds a multilevel header
    meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Std). """
    func_tri = {}
    for i in header[2:]:
        func_tri[i] = np.mean, np.std
    ABC_trips_avg = pd.pivot_table(triplicates, values=header[2:], index=["Master Protein Descriptions", "peptide-phosphosite"], aggfunc=func_tri)
    ABC_trips_avg = ABC_trips_avg.reset_index()[header]
    return ABC_trips_avg


def FilterByRange(X, rangeCut=0.4):
    """ Filter rows for those containing more than a range threshold. """
    Rg = X.iloc[:, X.columns.get_level_values(1) == "ptp"]
    Xidx = np.all(Rg.values <= rangeCut, axis=1)
    return X.iloc[Xidx, :]


def FilterByStdev(X, stdCut=0.5):
    """ Filter rows for those containing more than a standard deviation threshold. """
    XX = LinearFoldChange(X.copy())
    Stds = XX.iloc[:, XX.columns.get_level_values(1) == "std"]
    Xidx = np.all(Stds.values <= stdCut, axis=1)
    Means = pd.concat([X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, X.columns.get_level_values(1) == "mean"]], axis=1)
    return Means.iloc[Xidx, :]
