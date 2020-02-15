"""
This creates Figure 3.
"""
import os
import pandas as pd
import numpy as np
import scipy as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .common import subplotLabel, getSetup
from sklearn.model_selection import cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from msresist.parameter_tuning import MSclusPLSR_tuning, kmeansPLSR_tuning
from msresist.plsr import Q2Y_across_components, R2Y_across_components
from msresist.clustering import MassSpecClustering
from msresist.sequence_analysis import preprocess_seqs
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
from ..sequence_analysis import FormatName, pYmotifs
from ..pre_processing import preprocessing, MapOverlappingPeptides, BuildMatrix, TripsMeanAndStd, MergeDfbyMean
from ..figures.figure1 import plotpca_ScoresLoadings, plotVarReplicates

path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (3, 3))

    # blank out first axis for cartoon
    # ax[0].axis('off')

    # Read in data
    Y_cv1 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/GrowthFactors/CV_raw3.csv')).iloc[:30, :11]
    Y_cv2 = pd.read_csv(os.path.join(path, '../data/Phenotypic_data/GrowthFactors/CV_raw4.csv')).iloc[:29, :11]

    # Assert that there's no significant influence of the initial seeding density
    t = 72
    cv1t0 = Y_cv1[Y_cv1["Elapsed"] == 0].iloc[0, 1:]
    cv1 = Y_cv1[Y_cv1["Elapsed"] == t].iloc[0, 1:] / cv1t0

    cv2t0 = Y_cv2[Y_cv2["Elapsed"] == 0].iloc[0, 1:]
    cv2 = Y_cv2[Y_cv2["Elapsed"] == t].iloc[0, 1:] / cv2t0

    assert sp.stats.pearsonr(cv1t0, cv1)[1] > 0.05
    assert sp.stats.pearsonr(cv2t0, cv2)[1] > 0.05
    
    
    # Phosphorylation data
    X = preprocessing(AXLwt=True, motifs=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    X = preprocess_seqs(X, "Y").sort_values(by="Protein")

    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    treatments = list(d.index)

    # A: Cell Viability Endpoint
    BarPlot_FCendpoint(ax[0], cv1, cv2, t, list(Y_cv1.columns[1:]))

    # B: blank out second axis for signaling ClusterMap
    ax[1].axis('off')

    # C&D: Scores and Loadings MS data
    plotpca_ScoresLoadings(ax[2:4], d)

    # E: Variability across overlapping peptides in MS replicates
    X = preprocessing(AXLwt=True, rawdata=True)
    plotVarReplicates(ax[4:6], X)
    
    # F: AXL
    f = preprocessing(AXLwt=True, motifs=True, Vfilter=False, FCfilter=False, log2T=False, FCtoUT=True, mc_row=True)
    z = f.set_index(['Abbv', 'Sequence'])
    AXL(ax[6], z)

    # G: EGFR
    EGFR(ax[7], z)

    # H: Other RTKs
    OtherRTKs(ax[8], z)
    
    # I: ERK
    ERK(ax[9], z)

    # Add subplot labels
    subplotLabel(ax)

    return f


def BarPlot_FCendpoint(ax, x, y, t, lines):
    c = pd.concat([x, y]).reset_index()
    c.columns = ["lines", "Cell Viability (fold-change t=0)"]

    ax = sns.barplot(x="lines", 
                     y="Cell Viability (fold-change t=0)", 
                     data=c, ci="sd", 
                     color="darkcyan")

    ax.set_title("t =" + str(t) + "h")
    

def AXL(ax, f):
    axl481 = f.loc["AXL", "KKETRyGEVFE"].iloc[0, 3:]
    axl702 = f.loc["AXL", "IYNGDyYRQGR"].iloc[0, 3:]
    axl703 = f.loc["AXL", "YNGDYyRQGRI"].iloc[0, 3:]
    axl759 = f.loc["AXL", "ENSEIyDYLRQ"].iloc[0, 3:]
    axl866 = f.loc["AXL", "HPAGRyVLCPS"].iloc[0, 3:]

    ax.plot(axl702, marker="o", label="Y702-p", color = "darkorange")
    ax.plot(axl703, marker="o", label="Y703-p", color = "darkred")
    ax.plot(axl481, marker="o", label="Y481-p", color = "darkcyan")
    ax.plot(axl759, marker="o", label="Y759-p", color = "darkgreen")
    ax.plot(axl866, marker="o", label="Y866-p", color = "darkblue")
    ax.legend(loc=0)

    ax.set_xticklabels(list(axl702.index), rotation = 45)
    ax.set_ylabel("Normalized Signal")
    ax.set_title("AXL")


def EGFR(ax, f):
    egfr1172 = f.loc["EGFR", "LDNPDyQQDFF"].iloc[0, 3:]
    egfr1197 = f.loc["EGFR", "AENAEyLRVAP"].iloc[0, 3:]
    egfr1069 = f.loc["EGFR", "SFLQRySSDPT"].iloc[0, 3:]
    egfr1110 = f.loc["EGFR", "VQNPVyHNQPL"].iloc[0, 3:]
    egfr1131 = f.loc["EGFR", "QDPHStAVGNP"].iloc[0, 3:]

    ax.plot(egfr1172, marker="o", label="Y1172-p", color = "gray")
    ax.plot(egfr1197, marker="o", label="Y1197-p", color = "darkorange")
    ax.plot(egfr1069, marker="o", label="Y1069-p", color = "darkred")
    ax.plot(egfr1110, marker="o", label="Y1110-p", color = "darkgreen")
    ax.plot(egfr1131, marker="o", label="Y1131-p", color = "darkblue")
    ax.legend(loc=0)

    ax.set_xticklabels(list(egfr1172.index), rotation = 45)
    ax.set_title("EGFR")
    ax.set_ylabel("normalized signal")


def OtherRTKs(ax, f):
    met1003 = f.loc["MET", "NESVDyRATFP"].iloc[0, 3:]
    erbb2877 = f.loc["ERBB2", "IDETEyHADGG"].iloc[0, 3:]
    erbb31328 = f.loc["ERBB3", "FDNPDyWHSRL"].iloc[0, 3:]

    epha2_575 = f.loc["EPHA2", "SPEDVyFSKSE"].iloc[0, 3:]
    efnb2_304 = f.loc["EFNB2", "VFCPHyEKVSG"].iloc[0, 3:]

    epha1_594 = f.loc["EPHB1", "PGMKIyIDPFT"].iloc[0, 3:]
    efnb1_317 = f.loc["EFNB1", "NYCPHyEKVSG"].iloc[0, 3:]

    epha6_644 = f.loc["EPHB6", "GLGVKyyIDPS"].iloc[0, 3:]

    ax.plot(met1003, marker="o", label="MET Y1003-p", color = "black")
    ax.plot(erbb2877, marker="o", label="HER2 Y877-p", color = "darkorange")
    ax.plot(erbb31328, marker="o", label="HER3 Y1328-p", color = "darkred")
    ax.plot(epha1_594, marker="o", label="EPHA1 Y594-p", color = "darkblue")
    ax.plot(epha2_575, marker="o", label="EPHA2 Y575-p", color = "darkgreen")
    ax.plot(epha6_644, marker="o", label="EPHA6 Y644-p", color = "darkcyan")
    ax.legend(loc=0)

    ax.set_xticklabels(list(met1003.index), rotation = 45)
    ax.set_ylabel("Normalized Signal")
    ax.set_title("Other Bypass RTKs")


def adapters(ax, f):
    E_shb246 = f.loc["SHB", "TIADDySDPFD"].iloc[0, 3:]
    E_stat3686 = f.loc["STAT3", "EAFGKyCRPES"].iloc[0, 3:]
    E_shc1427 = f.loc["SHC1", "FDDPSyVNVQN"].iloc[0, 3:]
    E_gab1659 = f.loc["GAB1", "DERVDyVVVDQ"].iloc[0, 3:]
    E_gab2266 = f.loc["GAB2", "FRDSTyDLPRS"].iloc[0, 3:]
    E_crk136 = f.loc["CRK", "QEEAEyVRALF"].iloc[0, 3:]
    E_anxa2238 = f.loc["ANXA2", "KsYSPyDMLES"].iloc[0, 3:]

    ax.plot(E_shb246, marker="x", color="black", label="SHB Y246-p")
    ax.plot(E_stat3686, marker="x", color="darkorange", label="STAT3 Y686-p")
    ax.plot(E_shc1427, marker="x", color="darkred", label="SHC1 Y427-p")
    ax.plot(E_gab1659, marker="x", color="darkblue", label="GAB1 Y659-p")
    ax.plot(E_gab2266, marker="x", color="lightblue", label="GAB2 Y266-p")
    ax.plot(E_crk136, marker="x", color="darkgreen", label="CRK Y136-p")
    ax.plot(E_anxa2238, marker="x", color="darkcyan", label="ANXA2 238-p")

    ax.set_xticklabels(list(E_shb246.index), rotation = 45)
    ax.set_ylabel("Normalized Signal")
    ax.set_ylabel("Normalized Signal")

def ERK(ax, f):
    E_erk1s = f.loc["MAPK3", "GFLTEyVATRW"].iloc[0, 3:]
    E_erk1d = f.loc["MAPK3", "GFLtEyVATRW"].iloc[0, 3:]
    E_erk3s = f.loc["MAPK1", "GFLTEyVATRW"].iloc[0, 3:]
    E_erk3d = f.loc["MAPK1", "GFLtEyVATRW"].iloc[0, 3:]
    E_erk5s = f.loc["MAPK7", "YFMTEyVATRW"].iloc[0, 3:]

    ax.plot(E_erk1s, marker="o", color="black", label="ERK1 Y204-p")
#     ax.plot(E_erk1d, marker="o", color="darkorange", label="ERK1 T202-p;Y204-p")
    ax.plot(E_erk3s, marker="o", color="darkred", label="ERK3 Y187-p")
#     ax.plot(E_erk3d, marker="o", color="darkblue", label="ERK3 Y187-p;T185-p")
    ax.plot(E_erk5s, marker="o", color="darkgreen", label="ERK5 Y221-p")
    ax.legend(loc=0)

    ax.set_xticklabels(list(E_erk1s.index), rotation = 45)
    ax.set_ylabel("Normalized Signal")


def JNK(ax, f):
    E_jnk2_185s = f.loc["MAPK9", "FMMTPyVVTRY"].iloc[0, 3:]
    E_jnk2_223s = f.loc["MAPK10", "FMMTPyVVTRY"].iloc[0, 3:]

    ax.plot(E_jnk2_185s, marker="o", color="black", label="JNK2 Y185-p")
    ax.plot(E_jnk2_223s, marker="o", color="darkorange", label="JNK3 Y223-p")
    ax.legend(loc=0)

    ax.set_xticklabels(list(E_jnk2_185s.index), rotation = 45)
    ax.set_ylabel("Normalized Signal")


def P38(ax, f):
    E_p38G_185 = f.loc["MAPK12", "SEMTGyVVTRW"].iloc[0, 3:]
    E_p38D_182 = f.loc["MAPK13", "AEMTGyVVTRW"].iloc[0, 3:]
    E_p38A_182 = f.loc["MAPK14", "DEMTGyVATRW"].iloc[0, 3:]

    ax.plot(E_p38G_185, marker="o", color="darkred", label="P38G Y185-p")
    ax.plot(E_p38D_182, marker="o", color="darkblue", label="P38D Y182-p;T185-p")
    ax.plot(E_p38A_182, marker="o", color="darkgreen", label="P38A Y182-p")
    ax.legend(loc=0)

    ax.set_xticklabels(list(E_p38G_185.index), rotation = 45)
    ax.set_ylabel("Normalized Signal")
