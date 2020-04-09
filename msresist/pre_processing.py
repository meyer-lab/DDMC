""" This scripts handles all the pre-processing required to merge and transform the raw mass spec biological replicates. """

import os
import numpy as np
import pandas as pd
from scipy import stats
from .sequence_analysis import FormatName, pYmotifs


path = os.path.dirname(os.path.abspath(__file__))


###-------------------------- Pre-processing MS data --------------------------###
def preprocessing(
        AXLwt=False,
        Axlmuts_Erl=False,
        Axlmuts_ErlAF154=False,
        Axlmuts_ErlAF154_BR2=False,
        CPTAC=False,
        Vfilter=False,
        FCfilter=False,
        log2T=False,
        FCtoUT=False,
        rawdata=False,
        mc_row=True,
        mc_col=False):
    """ Input: Raw MS bio-replicates. Output: Mean-centered merged data set.
    1. Concatenation, 2. log-2 transformation, 3. Mean-Center, 4. Merging, 5. Fold-change,
    6. Filters: 'Vfilter' filters by correlation when 2 overlapping peptides or std cutoff if >= 3.
    Note 1: 'motifs' redefines peptide sequences as XXXXXyXXXXX which affects merging.
    Note 2: Data is converted back to linear scale before filtering so 'log2T=True' to use log-scale for analysis.
    Note 3: CPTAC is already normalized, so: mc_row and mc_col = False """
    filesin = list()

    if AXLwt:
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/20180817_JG_AM_TMT10plex_R1_psms_raw.csv"), header=0))
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/20190214_JG_AM_PC9_AXL_TMT10_AC28_R2_PSMs_raw.csv"), header=0))
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/CombinedBR3_TR1&2_raw.csv"), header=0))
    if Axlmuts_Erl:
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_unstim_BR1_raw.csv"), header=0))
    if Axlmuts_ErlAF154:
        br1 = pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_ActivatingAb_BR1_raw.csv"), header=0)
        br2 = pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_ActivatingAb_BR2_raw.csv"))
        br2.columns = br1.columns
        filesin.append(br1)
        filesin.append(br2)
    if CPTAC:
        X = preprocessCPTAC()
        filesin.append(X)

    data_headers = list(filesin[0].select_dtypes(include=['float64']).columns)
    FCto = data_headers[0]
    
    if mc_row or mc_col:
        X = MeanCenter(Log2T(pd.concat(filesin), data_headers), data_headers, mc_row, mc_col)
        fullnames, genes = FormatName(X)
        X["Protein"] = fullnames
        X.insert(3, "Gene", genes)
        merging_indices = list(X.columns[:4])
    else:
        X = pd.concat(filesin)
        genes = list(X["Gene"])
        merging_indices = list(X.columns[:3])

    if rawdata:
        return X

    X = pYmotifs(X, genes)
    merging_indices.insert(3, "Position")

    if Vfilter:
        X = VFilter(X, merging_indices, data_headers, FCto)

    object_headers = list(X.select_dtypes(include=['object']).columns)

    X = MergeDfbyMean(
        X.copy(),
        data_headers,
        merging_indices
        ).reset_index()[object_headers + data_headers]

    if FCfilter:
        X = FoldChangeFilter(X, data_headers, FCto)

    if not log2T:
        if FCtoUT:
            X = LinearFoldChange(X, data_headers, FCto)
        if not FCtoUT:
            X = Linear(X, data_headers)

    return X


def preprocessCPTAC():
    """ Replace patient identifiers, fill NaNs, and make it compatible with current code. """
    X = pd.read_csv(os.path.join(path, "./data/MS/CPTAC/CPTAC3_Lung_Adeno_Carcinoma_Phosphoproteome.phosphopeptide.tmt10.csv"))
    d = X.iloc[:, 1:-3]
    X = pd.concat([X.iloc[:, 0], X.iloc[:, -3:], d.loc[:, d.columns.str.contains("CPT")]], axis=1)

    n = pd.read_csv(os.path.join(path, "./data/MS/CPTAC/S046_BI_CPTAC3_LUAD_Discovery_Cohort_Samples_r1_May2019.csv"))
    bi_id = list(n[~n["Broad Sample.ID"].str.contains("IR")].iloc[:, 1])
    X.columns = ["Sequence"] + list(X.columns[1:4]) + bi_id

    return X.drop("Organism", axis=1)


def filter_NaNpeptides(X):
    Xidx = np.count_nonzero(~np.isnan(X.iloc[:, 4:]), axis=1) / X.iloc[:, 4:].shape[1] >= 0.15
    return X.iloc[Xidx, :]


def MergeDfbyMean(X, values, indices):
    """ Compute mean across duplicates. """
    return pd.pivot_table(X, values=values, index=indices, aggfunc=np.mean)


def LinearFoldChange(X, data_headers, FCto):
    """ Convert to linear fold-change from log2 mean-centered. """
    X[data_headers] = pd.DataFrame(np.power(2, X[data_headers])).div(np.power(2, X[FCto]), axis=0)
    return X


def Linear(X, data_headers):
    """ Convert to linear fold-change from log2 mean-centered. """
    X[data_headers] = pd.DataFrame(np.power(2, X[data_headers]))
    return X


def FoldChangeToControl(X, data_headers):
    """ Convert to fold-change to control. """
    X[data_headers] = X[data_headers].div(X.iloc[:, 3], axis=0)
    return X


def Log2T(X, data_headers):
    """ Convert to log2 scale keeping original sign. """
    X[data_headers] = np.log2(X[data_headers])
    return X


def MeanCenter(X, data_headers, mc_row, mc_col):
    """ Mean centers each row of values. logT also optionally log2-transforms. """
    if mc_row:
        X[data_headers] = X[data_headers].sub(X[data_headers].mean(axis=1), axis=0)
    if mc_col:
        X[data_headers] = X[data_headers].sub(X[data_headers].mean(axis=0), axis=1)
    return X


def VarianceFilter(X, data_headers, varCut=0.1):
    """ Filter rows for those containing more than cutoff variance. Variance across conditions per peptide.
    Note this should only be used with log-scaled, mean-centered data. """
    Xidx = np.var(X[data_headers].values, axis=1) > varCut
    return X.iloc[Xidx, :]  # .iloc keeps only those peptide labeled as "True"


def FoldChangeFilter(X, data_headers, FCto, cutoff=0.2):
    """ Filter rows for those containing more than a two-fold change.
    Note this should only be used with linear-scale data normalized to the control. """
    XX = LinearFoldChange(X.copy(), data_headers, FCto)
    Xidx = np.any(XX[data_headers].values <= 1 - cutoff, axis=1) | np.any(XX[data_headers].values >= 1 + cutoff, axis=1)
    return X.iloc[Xidx, :]


###------------ Filter by variance (stdev or range/pearson's) ------------------###


def VFilter(ABC, merging_indices, data_headers, FCto):
    """ Filter based on variability across recurrent peptides in MS biological replicates """
    NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

    NonRecTable = BuildMatrix(NonRecPeptides, ABC, data_headers, FCto)
    NonRecTable = NonRecTable.assign(BioReps=list("1" * NonRecTable.shape[0]))
    NonRecTable = NonRecTable.assign(r2_Std=list(["N/A"] * NonRecTable.shape[0]))

    CorrCoefPeptides = BuildMatrix(CorrCoefPeptides, ABC, data_headers, FCto)
    DupsTable = CorrCoefFilter(CorrCoefPeptides)
    DupsTable = MergeDfbyMean(DupsTable, DupsTable[data_headers], merging_indices)
    DupsTable = DupsTable.assign(BioReps=list("2" * DupsTable.shape[0])).reset_index()

    StdPeptides = BuildMatrix(StdPeptides, ABC, data_headers, FCto)
    TripsTable = TripsMeanAndStd(StdPeptides, merging_indices + ["BioReps"], data_headers)
    TripsTable = FilterByStdev(TripsTable)

    merging_indices.insert(4, "BioReps")
    merging_indices.insert(5, "r2_Std")

    ABC = pd.concat([
        NonRecTable,
        DupsTable,
        TripsTable
    ]
    ).reset_index()[merging_indices[:3] + list(ABC[data_headers].columns) + merging_indices[3:]]

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


def BuildMatrix(peptides, ABC, data_headers, FCto):
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
            fc = LinearFoldChange(pepts[data_headers].copy(), data_headers, FCto)
            corrcoef, _ = stats.pearsonr(fc.iloc[0, :], fc.iloc[1, :])
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


def CorrCoefFilter(X, corrCut=0.5):
    """ Filter rows for those containing more than a correlation threshold. """
    Xidx = X.iloc[:, -1].values >= corrCut
    return X.iloc[Xidx, :]


def TripsMeanAndStd(triplicates, merging_indices, data_headers):
    """ Merge all triplicates by mean and standard deviation across conditions. Note this builds a multilevel header
    meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Std). """
    func_tri = {}
    for i in triplicates[data_headers].columns:
        func_tri[i] = np.mean, np.std
    X = pd.pivot_table(triplicates, values=triplicates[data_headers].columns, index=merging_indices, aggfunc=func_tri)
    return X.reset_index()


def FilterByRange(X, rangeCut=0.4):
    """ Filter rows for those containing more than a range threshold. """
    Rg = X.iloc[:, X.columns.get_level_values(1) == "ptp"]
    Xidx = np.all(Rg.values <= rangeCut, axis=1)
    return X.iloc[Xidx, :]


def FilterByStdev(X, stdCut=0.5):
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


def peptidefinder(X, loc, Protein=False, Gene=False, Sequence=False):
    """ Search for a specific peptide either by name or sequence. """
    if Protein:
        found = X[X["Protein"].str.contains(loc)]
    if Gene:
        found = X[X["Gene"].str.contains(loc)]
    if Sequence:
        found = X[X["Sequence"].str.contains(loc)]
    return found


######----------pre-processing of phenotype data----------######

def MergeTR(data):
    """ Convenient to merge by mean all TRs of IncuCyte """
    for i in range(1, data.shape[1], 2):
        data.iloc[:, i] = data.iloc[:, i:i + 2].mean(axis=1)

    return data.drop(data.columns[[i + 1 for i in range(1, data.shape[1], 2)]], axis="columns")


def cv_pre(cv1, cv2, cv3, tr, itp, ftp, lines):
    """ Preprocesses cell viability data sets for analysis. """
    l = [cv1, cv2, cv3]
    z = []
    for i in range(len(l)):
        c = l[i].loc[:, l[i].columns.str.contains(tr)]
        c.insert(0, "Elapsed",  cv1.iloc[:, 0])
        fc = c[c["Elapsed"] == ftp].iloc[0, 1:].div(c[c["Elapsed"] == itp].iloc[0, 1:])
        z.append(fc)
    
    cv = pd.DataFrame(pd.concat(z, axis=0)).reset_index()
    cv.columns = ["lines", "viability"]
    cv = cv.groupby("lines").mean().T
    return cv[lines].iloc[0, :]


def cm_pre(X, tr, ftp, lines):
    """ Preprocesses migration data sets for analysis. """
    x = X.loc[:, X.columns.str.contains(tr)]
    x.insert(0, "Elapsed", X.iloc[:, 0])
    cm = x[x["Elapsed"] == ftp].iloc[0, 1:]
    return cm[lines]

