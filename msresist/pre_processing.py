import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msresist.sequence_analysis import GeneratingKinaseMotifs


###-------------------------- Pre-processing Raw Data --------------------------###

def preprocessing(A_r, B_r, C_r, treatments, motifs=False, FCfilter=False, logT=False):
    ABC_mc = pd.concat([A_r, B_r, C_r])
    ABC_mc = MeanCenter(ABC_mc, logT=False)
    
    if motifs:
        #Temporary: deleting peptides breaking the code
        ABC_mc = ABC_mc[ABC_mc["peptide-phosphosite"] != 'sEQLkPLktYVDPHTYEDPNQAVLk-1']
        ABC_mc = ABC_mc[ABC_mc["peptide-phosphosite"] != 'tYVDPHTYEDPNQAVLk-1']
        ABC_mc = ABC_mc[ABC_mc["peptide-phosphosite"] != 'tYELLNcDk-1']
        ABC_mc = ABC_mc[ABC_mc["peptide-phosphosite"] != 'sLYHDISGDTSGDYRk-1']
        ABC_mc = ABC_mc[ABC_mc["peptide-phosphosite"] != 'sYDVPPPPMEPDHPFYSNISk-1']
        
        ABC_names = FormatName(ABC_mc)
        ABC_seqs = FormatSeq(ABC_mc)

        directory = "./msresist/data/Sequence_analysis/"
        _, motifs = GeneratingKinaseMotifs(directory + "FaFile.fa", ABC_names, ABC_seqs, directory + "MatchedFaFile.fa", directory + "proteome_uniprot.fa")
        ABC_mc['peptide-phosphosite'] = motifs
    
    ABC_mc = MergeDfbyMean(ABC_mc, treatments)
    ABC_mc = ABC_mc.reset_index()[A_r.columns]
    ABC_mc = FoldChangeToControl(ABC_mc, logT=True)

    if FCfilter:
        ABC_mc = FoldChangeFilter(ABC_mc)
        
    if logT:
        ABC_mc.iloc[:, 2:] = np.sign(ABC_mc.iloc[:,2:]).multiply(np.log2(abs(ABC_mc.iloc[:,2:])), axis = 0)   #Model not predictive with this transformation

    return ABC_mc


def MergeDfbyMean(X, t):
    """ Compute mean across duplicates. """
    func = {}
    for i in t:
        func[i] = np.mean
    ABC_avg = pd.pivot_table(X, values=t, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func)
    return ABC_avg


def FoldChangeToControl(X, logT=False):
    """ Convert to fold-change to control """
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
    Xidx = np.var(X.iloc[:, 2:].values, axis=1) > varCut  # This uses booleans to determine if a peptides passes the filter "True" or not "False".
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
        if seq[0] != "y":
            seqs.append(seq[1:].split("-")[0])
        else:
            seqs.append(seq.split("-")[0])
    return seqs


def MapOverlappingPeptides(ABC):
    """ Find what peptides show up only 1, 2, or 3 times across all three replicates.
    Note that it's easier to create two independent files with each group to use aggfunc later. """
    dups = pd.pivot_table(ABC, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc="size").sort_values()
    dups_counter = {i: list(dups).count(i) for i in list(dups)}
    print("total number of recurrences:", dups_counter)
    dups = pd.DataFrame(dups)
    return dups


def BuildDupsMatrix(DupPeptides, ABC):
    """ Using the df of peptides (names and seqs) that show up in 2 replicates,
    build a matrix with all values across conditions for each peptide. Note 1: Nothing
    should go to the else statement. Note 2: there may be peptides labeled as '(blank)'
    which also show up in a different row correctly labeled. """
    dupslist = []  # should be shape 492
    for idx, dupseq in enumerate(DupPeptides.iloc[:, 1]):
        dup_name = DupPeptides.iloc[idx, 0]
        pepts = ABC.reset_index().set_index(["peptide-phosphosite", "Master Protein Descriptions"], drop=False).loc[dupseq, dup_name]
        names = pepts.iloc[:, 2]
        if dup_name == "(blank)":
            continue
        elif len(pepts) == 2 and len(set(names)) == 1:
            for i in range(len(pepts)):
                dupslist.append(pepts.iloc[i, :])
        else:
            print("check this")
            print(pepts)
    return pd.DataFrame(dupslist).reset_index(drop=True).iloc[:, 1:]


def BuildTripsMatrix(TripPeptides, ABC):
    """ Build matrix with all values across conditions for each peptide showing up in all 3 experients. """
    tripslist = []  # should be shape 492
    for idx, tripseq in enumerate(TripPeptides.iloc[:, 1]):
        trip_name = TripPeptides.iloc[idx, 0]
        pepts = ABC.reset_index().set_index(["peptide-phosphosite", "Master Protein Descriptions"], drop=False).loc[tripseq, trip_name]
        names = pepts.iloc[:, 2]
        seq = pepts.iloc[:, 1]
        if trip_name == "(blank)":
            continue
        if len(pepts) >= 3 and len(set(names)) == 1:
            for i in range(len(pepts)):
                tripslist.append(pepts.iloc[i, :])
        else:
            print("check this")
    return pd.DataFrame(tripslist).reset_index(drop=True).iloc[:, 1:]


def DupsMeanAndRange(duplicates, t, header):
    """ Merge all duplicates by mean and range across conditions. Note this means the header
    is multilevel meaning we have 2 values for each condition (eg within Erlotinib -> Mean | Reange). """
    func_dup = {}
    for i in t:
        func_dup[i] = np.mean, np.ptp  # np.corrcoef
    ABC_dups_avg = pd.pivot_table(duplicates, values=t, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func_dup)
    ABC_dups_avg = ABC_dups_avg.reset_index()[header]
    return ABC_dups_avg


def TripsMeanAndStd(triplicates, t, header):
    """ Merge all triplicates by mean and standard deviation across conditions. """
    func_tri = {}
    for i in t:
        func_tri[i] = np.mean, np.std
    ABC_trips_avg = pd.pivot_table(triplicates, values=t, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func_tri)
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
