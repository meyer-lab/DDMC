import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def MergeDfbyMean(A, B, C, t):
    # Concatenate 3 BR
    frames = [A, B, C]
    ABC = pd.concat(frames)
    print("shape of concatenated matrix:", ABC.shape)

    # count number of duplicates across data sets
    dups = pd.pivot_table(ABC, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc="size")
    my_dict = {i: list(dups).count(i) for i in list(dups)}
    print("total number of recurrences:", my_dict)

    # compute mean across duplicates
    func = {}
    for i in t:
        func[i] = np.mean
    ABC_avg = pd.pivot_table(ABC, values=t, index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc=func)

    dups2 = ABC_avg.pivot_table(index=['Master Protein Descriptions', 'peptide-phosphosite'], aggfunc="size")  # Add assertion, size always 1. Substract shapes.
    print("shape of averaged matrix:", ABC_avg.shape)
    return ABC_avg


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
