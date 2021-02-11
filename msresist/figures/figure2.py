"""
This creates Figure 2.
"""

from .common import subplotLabel, getSetup
import pandas as pd
from msresist.pre_processing import preprocessing
from msresist.figures.figure1 import IndividualTimeCourses, barplot_UtErlAF154

pd.set_option("display.max_columns", 30)
endpointcolors = ["light grey", "dark grey", "light navy blue", "jungle green"]


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (2, 3), multz={8: 1, 10: 1})

    # Read in Cell Migration data on collagen
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    r1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR1_RWD.csv")
    r2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR2_RWD.csv")
    r3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR3_RWD.csv")
    r4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR4_RWD.csv")

    cols = []
    for label in r1.columns:
        if "UT" in label:
            cols.append(label.split(" ")[0])
        else:
            cols.append(label.replace(" ", "-"))

    r1.columns = cols
    r2.columns = cols
    r3.columns = cols
    r4.columns = cols

    ds = [r2, r3, r4]
    ftp = 24
    tr1 = ["UT", "AF", "-E", "A/E"]
    tr2 = ["Untreated", "AF154", "Erlotinib", "Erl + AF154"]
    ylabel = "Relative Wound Density"

    IndividualTimeCourses(ds, ftp, lines, tr1, tr2, ylabel, TimePointFC=False, TreatmentFC=False, plot="KI", ax_=ax[0])
    IndividualTimeCourses(ds, ftp, lines, tr1, tr2, ylabel, TimePointFC=False, TreatmentFC=False, plot="KO", ax_=ax[1])
    IndividualTimeCourses(ds, ftp, lines, tr1, tr2, ylabel, TimePointFC=False, TreatmentFC=False, plot="Y698F", ax_=ax[2])
    IndividualTimeCourses(ds, ftp, lines, tr1, tr2, ylabel, TimePointFC=False, TreatmentFC=False, plot="Y821F", ax_=ax[3])

    ftp = 10

    tr1 = ["UT", "AF", "-E", "A/E"]
    tr2 = ["Untreated", "AF154", "Erlotinib", "Erl + AF154"]

    c = ["white", "windows blue", "scarlet", "dark green"]
    barplot_UtErlAF154(ax[4], lines, ds, ftp, tr1, tr2, "RWD (%)", "Cell Migration (24h)", TimePointFC=False, TreatmentFC=False, colors=c)

    tr1 = ["AF", "-E", "A/E"]
    tr2 = ["AF154", "Erlotinib", "Erl + AF154"]
    barplot_UtErlAF154(ax[5], lines, ds, ftp, tr1, tr2, "RWD fc to UT", "Cell Migration (24h)", TimePointFC=False, TreatmentFC="-UT", colors=c)

    # Phosphorylation levels of selected peptides
    X = preprocessing(Axlmuts_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X.select_dtypes(include=['float64']).T

    all_lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    d.index = all_lines

    # set1 = {"BCAR1": "Y234-p", "BCAR3": "Y212-p", "NEDD9": ["189-p", "Y166-p"]}
    # plot_AllSites(ax[6], X.copy(), set1, "BCAR & NEDD")

    # Reduced Heatmaps
    #     hm_af154 = mpimg.imread('msresist/data/MS/AXL/CM_reducedHM_AF154.png')
    #     hm_erl = mpimg.imread('msresist/data/MS/AXL/CM_reducedHM_Erl.png')
    #     ax[8].imshow(hm_af154)
    #     ax[8].axis("off")
    #     ax[9].imshow(hm_erl)
    #     ax[9].axis("off")

    # Add subplot labels
    subplotLabel(ax)

    return f
