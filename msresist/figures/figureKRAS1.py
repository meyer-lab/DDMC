"""
This creates KRAS figure
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from scipy.stats import zscore
from sklearn.cross_decomposition import PLSRegression
from .common import subplotLabel, getSetup
from ..pre_processing import MeanCenter, Log2T
from ..validations import pos_to_motif
from ..clustering import MassSpecClustering
from .figure1 import plotPCA
from .figure2 import plotDistanceToUpstreamKinase, plotR2YQ2Y, plotScoresLoadings
from .figureM5 import plot_GO


sns.set(color_codes=True)

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Preprocess mass spec data
    X = pd.read_csv("msresist/data/MS/KRAS_G12C_Haura.csv")
    X = MeanCenter(X, mc_row=True, mc_col=False)
    X.insert(1, "Position", [(aa + pos).split(";")[0] for aa, pos in zip(X["Amino Acid"], X["Positions Within Proteins"])])
    X = X.drop(["Amino Acid", "Positions Within Proteins"], axis=1)
    motifs, del_ids = pos_to_motif(X["Gene"], X["Position"])
    X = X.set_index(["Gene", "Position"]).drop(del_ids).reset_index()
    X.insert(0, "Sequence", motifs)
    d = X.select_dtypes(include=[float]).T
    i = X.select_dtypes(include=[object])

    # Unpickle DDMC model and find clusters
    with open('msresist/data/pickled_models/KRAS_Haura_Binomial_CL15_W10', 'rb') as m:    
        model = pickle.load(m)

    centers = pd.DataFrame(model.transform()).T
    centers.columns = d.index
    centers.index = np.arange(model.ncl) + 1

    cols = centers.columns
    centers = centers.T
    centers["Cell Line"] = [i.split("_")[0] for i in cols]
    centers["Time point"] = [i.split("_")[1] for i in cols]

    #PCA
    plotPCA(ax[:2], centers.reset_index(), 2, ["Cell Line", "Time point"], "Cluster", hue_scores="Cell Line", style_scores="Time point")

    # Upstream Kinases
    cOI = [5, 9, 11, 14]
    plotDistanceToUpstreamKinase(model, cOI, ax[2], num_hits=4)

    # Cluster centers
    cData = pd.melt(frame=centers, id_vars=["Cell Line", "Time point"], value_vars=centers.columns[:-2], value_name="Center", var_name="Cluster")

    for i, c in enumerate(cOI):
        cD = cData[cData["Cluster"] == c]
        sns.lineplot(data=cD, x="Time point", y="Center", hue="Cell Line", ax=ax[i + 3])
        ax[i + 3].set_title("Center Cluster " + str(c))

    # C5 analysis
    """
    Includes 57 peptides that all show a striking increase in abundance after 24h compared with 0h and 6h in H358 cells (epithelial). 
    Includes a HER2 peptide (T1240, not reported in PsP). HER2 signaling is identified in Hitendra's paper as a key driver of resistance in epithelial cells. 
    - Is this a HER2 cluster? 
    - Could this include new HER2 substrates / signaling components? 
    - Maybe NEKs, activated by HER2, phosphorylates a bunch of these peptides?
    - Gene ontology analysis of these 57 peptides found no statistically significant biological processes
    """
    d = {"ERBB2": "T1240", "NEK7": "Y28", "ELMO2" : "S94", "TCP11": "S385", "CCDC6": "S395", "RNF139": "Y577"}
    plot_IdSites(ax[6], X, d, "", rn=False, ylim=False, xlabels=X.columns[3:-1])
    ax[6].set_xticklabels(X.columns[3:-1], rotation=90)
    ax[6].set_title("Cluster 5 peptides")

    plot_GO(5, ax[7], n=7, title="GO Cluster 5", max_width=20, analysis="KRAS")
    ax[7].set_title("GO Cluster 5 peptides")

    d = {"AJUBA": "T114", "SHANK2": "S65", "SHANK2": "S41", "LIMD1": "S314", "DLG5": "S1263", "DLG5": "S1700", "WWC3": "T909", "MARK3": "S598", "MARK3": "S42", "MARK3": "S601"}
    plot_IdSites(ax[8], X, d, "Negative Hippo peptides", rn=False, ylim=False, xlabels=X.columns[3:-1])
    ax[8].set_xticklabels(X.columns[3:-1], rotation=90)

    # C14 analysis
    """
    Small cluster (151 peptides) with mainly p-sites whose genes regulate MT-based cytoskeletal reorganization and cell cycle. 
    Interstingly, NEK7 is one of the kinases predicted to be upstream of this cluster and is present, showing the general trend of "attenuating-down". 
    Same GO results using complete H358 portion as with entire data set
    """
    plot_GO(14, ax[9], n=7, title="GO Cluster 14", max_width=20, analysis="KRAS")

    d = {"ATRX": "T674", "NEK7": "T30", "MARK3": "T564", "CENPC": "S528", "SMC3": "S1074", "CCND1": "T286"}
    plot_IdSites(ax[10], X, d, "Microtubule Cytoskleleton Organization", rn=False, ylim=False, xlabels=X.columns[3:-1])
    ax[10].set_xticklabels(X.columns[3:-1], rotation=90)
    ax[10].set_title("Cell Cycle & MT Cytoskleleton Organization")

    # C9 analysis
    plot_GO(9, ax[11], n=7, title="GO Cluster 9", max_width=40, analysis="KRAS")

    d1 = {"NIPBL": "S350", "CDCA5": "S21", "RB1": "S249", "WAPL": "S226"}
    plot_IdSites(ax[12], X, d1, "", rn=False, ylim=False, xlabels=X.columns[3:-1])
    ax[12].set_xticklabels(X.columns[3:-1], rotation=90)
    ax[12].set_title("Cohesin Loading")

    d2 = {"DOCK7": "S30", "HOOK3": "S238", "PCM1": "S119"}
    plot_IdSites(ax[13], X, d2, "", rn=False, ylim=False, xlabels=X.columns[3:-1])
    ax[13].set_xticklabels(X.columns[3:-1], rotation=90)
    ax[13].set_title("Interkinetic nuclear migration")

    #C11 analysis
    plot_GO(11, ax[14], n=7, title="GO Cluster 11", max_width=40, analysis="KRAS")

    plot_GO("11_H1792", ax[15], n=7, title="GO Cluster 11", max_width=40, analysis="KRAS")
    ax[15].set_title("Cluster 11 with H1792 peptides complete")

    d = {"SRC": "S12", "CDC6": "S106", "TICRR": "T1633", "ORC1": "S273", "CLSPN": "S1289"}
    plot_IdSites(ax[16], X, d, "", rn=False, ylim=False, xlabels=X.columns[3:-1])
    ax[16].set_xticklabels(X.columns[3:-1], rotation=90)
    ax[16].set_title("Mitotic DNA replication checkpoint signaling")

    return f


def plot_IdSites(ax, x, d, title=False, rn=False, ylim=False, xlabels=False):
    """ Plot a set of specified p-sites. 'd' should be a dictionary werein every item is a protein-position pair. """
    n = list(d.keys())
    p = list(d.values())
    dfs = []
    for i in range(len(n)):
        x1 = x[(x["Gene"] == n[i]) & (x["Position"] == p[i])]
        dfs.append(x1.set_index(["Gene", "Position"]).select_dtypes(include=float))

    df = pd.concat(dfs)

    if rn:
        df = df.reset_index()
        df["Gene"] = rn
        df = df.set_index(["Gene", "Position"])

    data = pd.melt(frame=df.reset_index(), id_vars=["Gene", "Position"], value_vars=df.columns, var_name="Line", value_name="p-signal")

    data["Peptide"] = [g + ": " + p for g, p in zip(data["Gene"], data["Position"])]

    ax = sns.lineplot(x="Line", y="p-signal", data=data, hue="Peptide", ax=ax)

    if title:
        ax.set_title(title)
