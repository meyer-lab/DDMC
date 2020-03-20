"""
This creates Figure 4.
"""
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 9), (4, 3))

    # blank out first axis for cartoon
    ax[0].axis('off')

    # Add subplot labels
    subplotLabel(ax)

    return f
