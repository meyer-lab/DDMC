
import seaborn as sns
from .common import subplotLabel, getSetup
from msresist.figures.figure2 import plotCenters, plotMotifs
from ..clustering import MassSpecClustering
from ..pre_processing import preprocessing


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 5), (2, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load DDMC
    X = preprocessing(AXLwt_GF=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    data = X.select_dtypes(include=['float64']).T
    info = X.select_dtypes(include=['object'])
    model = MassSpecClustering(info, 5, SeqWeight=2, distance_method="PAM250").fit(X=data)
    lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    # Centers
    plotCenters(ax[:5], model, lines)

    # Plot motifs
    pssms, _ = model.pssms(PsP_background=True)
    plotMotifs([pssms[0], pssms[1], pssms[2], pssms[3], pssms[4]], axes=ax[5:10], titles=["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"], yaxis=[0, 11])

    return f
