"""
This creates Supplemental Figure 1: Cell Viability time course
"""

import seaborn as sns
from .common import subplotLabel, getSetup
from .figure1 import IndividualTimeCourses, import_phenotype_data


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 6), (2, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    cv = import_phenotype_data(phenotype="Cell Viability")
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]

    for i, line in enumerate(lines):
        IndividualTimeCourses(cv, 96, lines, tr1, tr2, "fold-change confluency", TimePointFC=24, TreatmentFC="-E", plot=line, ax_=ax[i], ylim=[0.8, 3.5])

    return f