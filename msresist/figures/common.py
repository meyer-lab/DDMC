"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_uppercase
from matplotlib import gridspec, pyplot as plt
import seaborn as sns


def getSetup(figsize, gridd):
    """ Establish figure set-up with subplots. """
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    ax = list()
    for x in range(gridd[0] * gridd[1]):
        ax.append(f.add_subplot(gs1[x]))

    return (ax, f)


def subplotLabel(axs):
    """ Place subplot labels on the list of axes. """
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.2, ascii_uppercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
