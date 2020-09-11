""" This scripts handles all the pre-processing required to merge and transform the raw mass spec biological replicates. """

import os
import numpy as np
import pandas as pd
from scipy import stats
from .motifs import FormatName, MapMotifs


path = os.path.dirname(os.path.abspath(__file__))


###-------------------------- Pre-processing MS data --------------------------###
def preprocessing(
    AXLwt=False, Axlmuts_Erl=False, Axlmuts_ErlAF154=False, CPTAC=False, Vfilter=False, FCfilter=False, log2T=False, FCtoUT=False, rawdata=False, mc_row=True, mc_col=False,
):
    """ Input: Raw MS bio-replicates. Output: Mean-centered merged data set.
    1. Concatenation, 2. log-2 transformation, 3. Mean-Center, 4. Merging, 5. Fold-change,
    6. Filters: 'Vfilter' filters by correlation when 2 overlapping peptides or std cutoff if >= 3.
    Note 1: 'motifs' redefines peptide sequences as XXXXXyXXXXX which affects merging.
    Note 2: Data is converted back to linear scale before filtering so 'log2T=True' to use log-scale for analysis.
    Note 3: CPTAC is already normalized, so: mc_row and mc_col = False """
    filesin = list()

    if AXLwt:
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/20180817_JG_AM_TMT10plex_R1_psms_raw.csv")))
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/20190214_JG_AM_PC9_AXL_TMT10_AC28_R2_PSMs_raw.csv")))
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/CombinedBR3_TR1&2_raw.csv")))
    if Axlmuts_Erl:
        filesin.append(pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_unstim_BR1_raw.csv")))
    if Axlmuts_ErlAF154:
        br1 = pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_ActivatingAb_BR1_raw.csv"))
        br2 = pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_ActivatingAb_BR2_raw.csv")).drop("UniprotAcc", axis=1)
        br2.columns = br1.columns
        br3 = pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_ActivatingAb_BR3_raw.csv"))
        br4 = pd.read_csv(os.path.join(path, "./data/MS/AXL/PC9_mutants_ActivatingAb_BR4_raw.csv"))
        filesin.append(br1)
#         filesin.append(br2)
        filesin.append(br3)
        filesin.append(br4)
    if CPTAC:
        X = preprocessCPTAC()
        filesin.append(X)

    data_headers = list(filesin[0].select_dtypes(include=["float64"]).columns)
    FCto = data_headers[1]

    if mc_row or mc_col:
        X = MeanCenter(Log2T(pd.concat(filesin), data_headers), data_headers, mc_row, mc_col)
        fullnames, genes = FormatName(X)
        X["Protein"] = fullnames
        X.insert(3, "Gene", genes)
        merging_indices = list(X.select_dtypes(include=["object"]).columns)
    else:
        X = pd.concat(filesin)
        genes = list(X["Gene"])
        merging_indices = list(X.select_dtypes(include=["object"]).columns)

    if rawdata:
        return X

    X = MapMotifs(X, genes)
    merging_indices.insert(3, "Position")

    if Vfilter:
        X = VFilter(X, merging_indices, data_headers)

    X = MergeDfbyMean(X.copy(), data_headers, merging_indices).reset_index()[merging_indices + data_headers]

    if FCfilter:
        X = FoldChangeFilterBasedOnMaxFC(X, data_headers, cutoff=0.40)
#         X = FoldChangeFilterToControl(X, data_headers, FCto, cutoff=0.4)

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
    X = filter_NaNpeptides(X)

    n = pd.read_csv(os.path.join(path, "./data/MS/CPTAC/S046_BI_CPTAC3_LUAD_Discovery_Cohort_Samples_r1_May2019.csv"))
    bi_id = list(n[~n["Broad Sample.ID"].str.contains("IR")].iloc[:, 1])
    X.columns = ["Sequence"] + list(X.columns[1:4]) + bi_id

    return X.drop("Organism", axis=1)


def filter_NaNpeptides(X, cut=0.2):
    """ Filter peptides that have a given percentage of missingness """
    d = X.select_dtypes(include=["float64"])
    Xidx = np.count_nonzero(~np.isnan(d), axis=1) / d.shape[1] >= cut
    return X.iloc[Xidx, :]


def MergeDfbyMean(X, values, indices):
    """ Compute mean across duplicates. """
    return pd.pivot_table(X, values=values, index=indices, aggfunc=np.mean)


def LinearFoldChange(X, data_headers, FCto):
    """ Convert to linear fold-change from log2 mean-centered. """
    X[data_headers] = pd.DataFrame(np.power(2, X[data_headers])).div(np.power(2, X[FCto]), axis=0)
    return X


def Linear(X, data_headers):
    """ Convert to linear from log2 mean-centered. """
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
    return X.iloc[Xidx, :]


def FoldChangeFilterToControl(X, data_headers, FCto, cutoff=0.4):
    """ Filter rows for those containing more than a two-fold change.
    Note this should only be used with linear-scale data normalized to the control. """
    XX = LinearFoldChange(X.copy(), data_headers, FCto)
    Xidx = np.any(XX[data_headers].values <= 1 - cutoff, axis=1) | np.any(XX[data_headers].values >= 1 + cutoff, axis=1)
    return X.iloc[Xidx, :]


def FoldChangeFilterBasedOnMaxFC(X, data_headers, cutoff=0.5):
    """ Filter rows for those containing an cutoff% change of the maximum vs minimum fold-change
    across every condition. """
    XX = Linear(X.copy(), data_headers)
    X_ToMin = XX[data_headers] / XX[data_headers].min(axis=0)
    Xidx = np.any(X_ToMin.values >= X_ToMin.max().values * cutoff, axis=1)
    return X.iloc[Xidx, :]


###------------ Filter by variance (stdev or range/pearson's) ------------------###


def VFilter(ABC, merging_indices, data_headers):
    """ Filter based on variability across recurrent peptides in MS biological replicates """
    NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

    NonRecTable = BuildMatrix(NonRecPeptides, ABC, data_headers)
    NonRecTable = NonRecTable.assign(BioReps=list("1" * NonRecTable.shape[0]))
    NonRecTable = NonRecTable.assign(r2_Std=list(["N/A"] * NonRecTable.shape[0]))

    CorrCoefPeptides = BuildMatrix(CorrCoefPeptides, ABC, data_headers)
    DupsTable = CorrCoefFilter(CorrCoefPeptides, corrCut=0.6)
    DupsTable = MergeDfbyMean(DupsTable, DupsTable[data_headers], merging_indices + ["r2_Std"])
    DupsTable = DupsTable.assign(BioReps=list("2" * DupsTable.shape[0])).reset_index()

    StdPeptides = BuildMatrix(StdPeptides, ABC, data_headers)
    TripsTable = TripsMeanAndStd(StdPeptides, merging_indices + ["BioReps"], data_headers)
    TripsTable = FilterByStdev(TripsTable, merging_indices + ["BioReps"], stdCut=0.4)

    merging_indices.insert(4, "BioReps")
    merging_indices.insert(5, "r2_Std")

    ABC = pd.concat(
        [NonRecTable, DupsTable, TripsTable]
    ).reset_index()[merging_indices[:2] + list(ABC[data_headers].columns) + merging_indices[2:]]

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


def BuildMatrix(peptides, ABC, data_headers):
    """ Map identified recurrent peptides to generate complete matrices with values.
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
            fc = Linear(pepts[data_headers].copy(), data_headers)
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


def CorrCoefFilter(X, corrCut=0.6):
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


def FilterByStdev(X, merging_indices, stdCut=0.4):
    """ Filter rows for those containing more than a standard deviation threshold. """
    Stds = X.iloc[:, X.columns.get_level_values(1) == "std"]
    StdMeans = list(np.round(Stds.mean(axis=1), decimals=2))
    Xidx = np.all(Stds.values <= stdCut, axis=1)
    Means = pd.concat([X[merging_indices], X.iloc[:, X.columns.get_level_values(1) == "mean"]], axis=1)
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
        data.iloc[:, i] = data.iloc[:, i: i + 2].mean(axis=1)

    return data.drop(data.columns[[i + 1 for i in range(1, data.shape[1], 2)]], axis="columns")


def y_pre(ds, tr, ftp, phenotype, all_lines, itp=False):
    """ Preprocesses cell phenotype data sets for analysis. """
    z = []
    for sl in ds:
        c = sl.loc[:, sl.columns.str.contains(tr)]
        c.insert(0, "Elapsed", ds[0].iloc[:, 0])
        c = c[list(c.columns[:3]) + [c.columns[4]] + [c.columns[3]] + list(c.columns[5:])]
        if not isinstance(itp, bool):
            c = c[c["Elapsed"] == ftp].iloc[0, 1:].div(c[c["Elapsed"] == itp].iloc[0, 1:])
        else:
            c = c[c["Elapsed"] == ftp].iloc[0, 1:]
        z.append(c)

    y = pd.DataFrame(pd.concat(z, axis=0)).reset_index()
    y.columns = ["Lines", phenotype]
    y = y.groupby("Lines").mean().T[c.index].T.reset_index()

    y["Lines"] = [s.split(tr)[0] for s in y.iloc[:, 0]]
    y["Treatment"] = tr

    if "-" in y["Lines"][1]:
        y["Lines"] = [s.split("-")[0] for s in y.iloc[:, 0]]

    y["Lines"] = all_lines
    return y[["Lines", "Treatment", phenotype]]


def FixColumnLabels(cv):
    """ Fix column labels to use pandas locators. """
    l = []
    for label in cv[0].columns:
        if "-" not in label and label != "Elapsed":
            l.append(label + "-UT")
        if "-" in label or label == "Elapsed":
            l.append(label)

    for d in cv:
        d.columns = l

    return cv
