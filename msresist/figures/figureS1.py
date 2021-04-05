"""
This creates Supplemental Figure 1: Cell Viability time course
"""

import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from .figure1 import IndividualTimeCourses, import_phenotype_data, barplot_UtErlAF154
from ..distances import BarPlotRipleysK, PlotRipleysK


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 20), (11, 4))

    # Add subplot labels
    # subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Read in phenotype data
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")

    # Labels
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    tr1 = ["-UT", "-E", "-A/E"]
    tr2 = ["Untreated", "Erlotinib", "Erl + AF154"]
    t1 = ["UT", "AF", "-E", "A/E"]
    t2 = ["Untreated", "AF154", "Erlotinib", "Erl + AF154"]
    colors = ["white", "windows blue", "scarlet"]
    mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
    all_lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]
    itp = 24

    # Bar plots
    barplot_UtErlAF154(ax[0], lines, cv, 96, tr1, tr2, "fold-change confluency", "Cell Viability (t=96h)", colors, TreatmentFC="-E", TimePointFC=itp, loc='upper right')
    barplot_UtErlAF154(ax[1], lines, red, 72, tr1, tr2, "fold-change YOYO+ cells", "Cell Death (t=72h)", TreatmentFC="-E", colors=colors, TimePointFC=itp, loc='lower center')
    barplot_UtErlAF154(ax[2], lines, sw, 14, tr1, tr2, "fold-change RWD", "Cell Migration (t=14h)", TreatmentFC="-E", colors=colors, TimePointFC=itp)
    BarPlotRipleysK(ax[3], '48hrs', mutants, lines, ['ut', 'e', 'ae'], tr2, 6, np.linspace(1.5, 14.67, 1), colors, TreatmentFC="Erlotinib", ylabel="fold-change K estimate")

    # All time courses
    idx = np.arange(5, 45, 4) - 1
    for i, line in enumerate(lines):
        IndividualTimeCourses(cv, 96, lines, tr1, tr2, "fold-change confluency", TimePointFC=24, TreatmentFC="-E", plot=line, ax_=ax[idx[i]], ylim=[0.8, 3.5])
        IndividualTimeCourses(red, 96, all_lines, tr1, tr2, "fold-change apoptosis (YOYO+)", TimePointFC=itp, plot=line, ax_=ax[idx[i]+1], ylim=[0, 13])
        IndividualTimeCourses(sw, 24, all_lines, t1, t2, "RWD %", plot=line, ax_=ax[idx[i]+2])
        PlotRipleysK('48hrs', mutants[i], ['ut', 'e', 'ae'], 6, ax=ax[idx[i]+3], title=line)

    return f