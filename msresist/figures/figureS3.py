"""
This creates Supplemental Figure 3: Cell Migration time course
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

    cm = import_phenotype_data(phenotype="Migration")
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    t1 = ["UT", "AF", "-E", "A/E"]
    t2 = ["Untreated", "AF154", "Erlotinib", "Erl + AF154"]

    for i, line in enumerate(lines):
        IndividualTimeCourses(cm, 24, lines, t1, t2, "fold-change RWD", plot=line, ax_=ax[i])

    return f