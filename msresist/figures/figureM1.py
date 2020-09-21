"""
This creates Figure M3.
"""

from .common import subplotLabel, getSetup
import matplotlib.image as mpimg


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))

    plotAlg(ax[0])

    return f

def plotAlg(ax):
    img = mpimg.imread('msresist/figures/Em_algo.png')
    return ax.imshow(img)
