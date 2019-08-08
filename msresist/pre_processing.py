import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from msresist.sequence_analysis import GeneratingKinaseMotifs


###-------------------------- Pre-processing Raw Data --------------------------###

def preprocessing(A_r, B_r, C_r, motifs=False, FCfilter=False, Vfilter=False, log2T=False):
    ABC = pd.concat([A_r, B_r, C_r])
    ABC = Log2T(ABC)
    ABC_mc = MeanCenter(ABC, logT=False)
    ABC_names = FormatName(ABC_mc)
    ABC_seqs = FormatSeq(ABC_mc)
    ABC_mc['peptide-phosphosite'] = ABC_seqs
    ABC_mc['Master Protein Descriptions'] = ABC_names   
    
    if motifs:
        directory = "./msresist/data/Sequence_analysis/"
        names, motifs = GeneratingKinaseMotifs(directory + "FaFile.fa", ABC_names, ABC_seqs, directory + "MatchedFaFile.fa", directory + "proteome_uniprot.fa")
        ABC_mc['peptide-phosphosite'] = motifs
        ABC_mc['Master Protein Descriptions'] = names
    
    
    ABC_merged = MergeDfbyMean(ABC_mc, A_r.columns[2:])
    ABC_merged = ABC_merged.reset_index()[A_r.columns]
    ABC_merged = LinearScale(ABC_merged)
    ABC_mc = FoldChangeToControl(ABC_merged)
    
    if FCfilter:
        ABC_mc = FoldChangeFilter(ABC_mc)
       
        ABC = LinearScale(ABC)
        ABC = FoldChangeToControl(ABC)
        ABC = FoldChangeFilter(ABC)
        
    if Vfilter:
        ABC = FoldChangeToControl(ABC)
        NonRecPeptides, CorrCoefPeptides, StdPeptides = MapOverlappingPeptides(ABC)

        NonRecTable = BuildMatrix(NonRecPeptides, ABC)
        
        CorrCoefPeptides = BuildMatrix(CorrCoefPeptides, ABC)
        DupsTable = CorrCoefFilter(CorrCoefPeptides)
        DupsTable = MergeDfbyMean(CorrCoefPeptides, DupsTable.columns[2:])
        DupsTable = DupsTable.reset_index()[A_r.columns]
        
        StdPeptides = BuildMatrix(StdPeptides, ABC)
        TripsTable = TripsMeanAndStd(StdPeptides, A_r.columns)
        TripsTable = FilterByStdev(TripsTable, A_r.columns)
                
        ABC_mc = pd.concat([NonRecTable, DupsTable, TripsTable])
        
    if log2T:
        ABC_mc = Log2T(ABC_mc)
     
    return ABC_mc


def MergeDfbyMean(X, t):
    """ Compute mean across duplicates. """
    func = {}
    for i in t:
        func[i] = np.mean
    ABC_avg = pd.pivot_table(X, values=t, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func)
    return ABC_avg

def LinearScale(X):
    """ Convert to linear scale from log2 scale. """
    X.iloc[:, 2:] = np.power(2, X.iloc[:, 2:])
    return X

def Log2T(X):
    """ Convert to log2 scale keeping original sing. """    
    X.iloc[:, 2:] = np.sign(X.iloc[:, 2:]).multiply(np.log2(abs(X.iloc[:, 2:])), axis = 0) 
    return X
    
    
def FoldChangeToControl(X):
    """ Convert to fold-change to control. """
    X.iloc[:, 2:] = X.iloc[:, 2:].div(X.iloc[:, 2], axis = 0)
    return X

def MeanCenter(X, logT=False):
    """ Mean centers each row of values. logT also optionally log2-transforms. """
    if logT:
        X.iloc[:, 2:] = np.log2(X.iloc[:, 2:].values)

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
    Xidx = np.any(X.iloc[:, 2:].values <= 0.5, axis=1) | np.any(X.iloc[:, 2:].values >= 2.0, axis=1)
    return X.iloc[Xidx, :]


def FormatName(X):
    """ Keep only the general protein name, without any other accession information """
    names = []
    x = list(map(lambda v: names.append(v.split("OS")[0]), X.iloc[:, 1]))
    return names


def FormatSeq(X):
    """ Found out the first letter of the seq gave problems while grouping by seq so I'm deleting it
    whenever is not a pY as well as the final -1/-2"""
    seqs = []
    for seq in list(X.iloc[:, 0]):
        if seq[0] == "y" or seq[0] == "t" or seq[0] == "s":
            seqs.append(seq.split("-")[0])
        else:
            seqs.append(seq[1:].split("-")[0])
    return seqs


###------------ Filter by variance (stdev or range/pearson's) ------------------###


def MapOverlappingPeptides(ABC):
    """ Find what peptides show up only 1, 2, or 3 times across all three replicates.
    Note that it's easier to create two independent files with each group to use aggfunc later. """
    dups = pd.pivot_table(ABC, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc="size").sort_values()
    dups_counter = {i: list(dups).count(i) for i in list(dups)}
    dups = pd.DataFrame(dups).reset_index()
    dups.columns = [ABC.columns[1], ABC.columns[0], "Recs"]
    NonRecPeptides = dups[dups["Recs"] == 1]
    RangePeptides = dups[dups["Recs"] == 2]
    StdPeptides = dups[dups["Recs"] >= 3]
    return NonRecPeptides, RangePeptides, StdPeptides


def BuildMatrix(peptides, ABC):
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
        matrix = matrix.assign(CorrCoefs = corrcoefs)
    
    else:
        matrix = pd.DataFrame(peptideslist).reset_index(drop=True)
        
    return matrix


def CorrCoefFilter(X, corrCut=0.6):
    """ Filter rows for those containing more then 0.6 correlation. 
    Note this should only be used with linear-scale data normalized to the control. """
    Xidx = X.iloc[:, 12].values >= corrCut
    return X.iloc[Xidx, :]  


def DupsMeanAndRange(duplicates, header):
    """ Merge all duplicates by mean and range across conditions. Note this means the header
    is multilevel meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Reange). """
    func_dup = {}
    for i in header[2:]:
        func_dup[i] = np.mean, np.ptp  
    ABC_dups_avg = pd.pivot_table(duplicates, values=header[2:], index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func_dup)
    ABC_dups_avg = ABC_dups_avg.reset_index()[header]
    return ABC_dups_avg
    

def TripsMeanAndStd(triplicates, header):
    """ Merge all triplicates by mean and standard deviation across conditions. """
    func_tri = {}
    for i in header[2:]:
        func_tri[i] = np.mean, np.std
    ABC_trips_avg = pd.pivot_table(triplicates, values=header[2:], index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func_tri)
    ABC_trips_avg = ABC_trips_avg.reset_index()[header]
    return ABC_trips_avg


def FilterByRange(ABC_dups_avg, header):
    """ Iterates across the df and filters by range. """
    ABC_dups_avg = ABC_dups_avg.set_index(['Master Protein Descriptions', 'peptide-phosphosite'])
    dups_final, protnames, seqs = [], [], []
    for i in range(ABC_dups_avg.shape[0]):
        ptp = ABC_dups_avg.iloc[i, ABC_dups_avg.columns.get_level_values(1) == 'ptp']
        mean = ABC_dups_avg.iloc[i, ABC_dups_avg.columns.get_level_values(1) == 'mean']
        seq = ABC_dups_avg.index[i][1]
        name = ABC_dups_avg.index[i][0]
        if all(v <= 0.6 for v in ptp):
            dups_final.append(mean)
            seqs.append(seq)
            protnames.append(name)

    # Concatenate lists into a pandas df
    dups_final = pd.DataFrame(dups_final).reset_index().iloc[:, 1:]

    frames = [pd.DataFrame(seqs), pd.DataFrame(protnames), dups_final]
    dups_final = pd.concat(frames, axis=1)
    dups_final.columns = header
    dups_final = dups_final.sort_values(by="Master Protein Descriptions")
    return dups_final


def FilterByStdev(ABC_trips_avg, header):
    """ Iterates across the df and filters by standard deviation. """
    ABC_trips_avg = ABC_trips_avg.set_index(['Master Protein Descriptions', 'peptide-phosphosite'])
    trips_final, protnames, seqs = [], [], []
    for i in range(ABC_trips_avg.shape[0]):
        std = ABC_trips_avg.iloc[i, ABC_trips_avg.columns.get_level_values(1) == 'std']
        mean = ABC_trips_avg.iloc[i, ABC_trips_avg.columns.get_level_values(1) == 'mean']
        seq = ABC_trips_avg.index[i][1]
        name = ABC_trips_avg.index[i][0]
        if all(v <= 0.3 for v in std):
            trips_final.append(mean)
            seqs.append(seq)
            protnames.append(name)

    # Concatenate lists into a pandas df
    trips_final = pd.DataFrame(trips_final).reset_index().iloc[:, 1:]

    frames = [pd.DataFrame(seqs), pd.DataFrame(protnames), trips_final]
    trips_final = pd.concat(frames, axis=1)
    trips_final.columns = header
    trips_final = trips_final.sort_values(by="Master Protein Descriptions")
    return trips_final


###-------------------------------------- Plotting Raw Data --------------------------------------###

def AvsBacrossCond(A, B, t):
    frames = [A, B]
    ConcDf = pd.concat(frames)
    dups = ConcDf[ConcDf.duplicated(['Master Protein Descriptions', 'peptide-phosphosite'], keep=False)].sort_values(by="Master Protein Descriptions")
    AB_nodups = dups.copy().iloc[:, 0].drop_duplicates()

    assert(AB_nodups.shape[0] == 0.5 * (dups.shape[0]))  # Assert that NodupsAB / 2 = dupsAB

    A, B = [], []
    dups.set_index("peptide-phosphosite", inplace=True)
    for i in AB_nodups:
        pepts = dups.loc[i]
        A.append(pepts.iloc[0, 0:12])
        B.append(pepts.iloc[1, 0:12])

    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    # B = pd.DataFrame(B).reset_index().set_index("Master Protein Descriptions"), if we wanted to plot by protein name

    data = []
    for i in range(1, 11):
        tup = (A.iloc[:, i], B.iloc[:, i])
        data.append(tup)

    fig, axs = plt.subplots(10, sharex=True, sharey=True, figsize=(10, 20))

    # label xticks with peptide sequences and manipulate space between ticks
    N = A.shape[0]
    plt.xticks(np.arange(N), ([str(i) for i in np.arange(N)]), rotation=90)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([w.get_window_extent().width for w in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    groups = t
    ax_i = np.arange(10)

    for data, group, i in zip(data, groups, ax_i):
        x, y = data
        axs[i].plot(x, 'r.--', alpha=0.7, label=group, linewidth=0.5)  # linestyle = ''
        axs[i].plot(y, 'bx--', alpha=0.7, linewidth=0.5)
        axs[i].legend(loc=0)
        axs[i].set_ylim([0, 2.5])
    return fig


def AvsBvsCacrossCond(A, B, C, t):
    frames = [A, B, C]
    ConcDf = pd.concat(frames)
    dups = ConcDf[ConcDf.duplicated(['Master Protein Descriptions', 'peptide-phosphosite'], keep=False)].sort_values(by="Master Protein Descriptions")
    ABC_nodups = dups.copy().iloc[:, 0].drop_duplicates()

#     assert(ABC_nodups.shape[0] == (dups.shape[0])/3)  #Assert that NodupsAB / 2 = dupsAB

    A, B, C = [], [], []
    dups.set_index("peptide-phosphosite", inplace=True)
    for i in ABC_nodups:
        pepts = dups.loc[i]
        if pepts.shape[0] == 2:
            continue
        if pepts.shape[0] == 3:
            A.append(pepts.iloc[0, 0:12])
            B.append(pepts.iloc[1, 0:12])
            C.append(pepts.iloc[2, 0:12])

    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    C = pd.DataFrame(C)

    data = []
    for i in range(1, 11):
        tup = (A.iloc[:, i], B.iloc[:, i], C.iloc[:, i])
        data.append(tup)

    fig, axs = plt.subplots(10, sharex=True, sharey=True, figsize=(10, 20))
    plt.xticks(np.arange(A.shape[0]), ([str(i) for i in np.arange(A.shape[0])]), rotation=90)

    # label xticks with peptide sequences and manipulate space between ticks
    N = A.shape[0]
    plt.xticks(np.arange(N), ([str(i) for i in np.arange(N)]), rotation=90)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([w.get_window_extent().width for w in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * N + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    groups = t
    ax_i = np.arange(10)

    for data, group, i in zip(data, groups, ax_i):
        x, y, z = data
        axs[i].plot(x, 'r.--', alpha=0.7, label=group, linewidth=0.5)  # linestyle = ''
        axs[i].plot(y, 'bx--', alpha=0.7, linewidth=0.5)
        axs[i].plot(z, 'k^--', alpha=0.7, linewidth=0.5)
        axs[i].legend(loc=2)
        axs[i].set_ylim([0, 2.5])
    return fig
