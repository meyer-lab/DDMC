"""
This file contains functions that are used in multiple figures.
"""
import sys
import time
from string import ascii_uppercase
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
import svgutils.transform as st


def getSetup(figsize, gridd, multz=None):
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """Place subplot labels on the list of axes."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_uppercase[ii],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )


def overlayCartoon(
    figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1, rotate=None
):
    """Add cartoon to a figure file."""

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale_x=scalee * scale_x, scale_y=scalee * scale_y)
    if rotate:
        cartoon.rotate(rotate, x, y)

    template.append(cartoon)
    template.save(figFile)


def genFigure():
    """Main figure generation function."""
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec(f"from sccp.figures.{nameOut} import makeFigure", globals())
    ff = makeFigure()

    if ff is not None:
        ff.savefig(
            f"./output/{nameOut}.svg", dpi=300, bbox_inches="tight", pad_inches=0
        )

    if sys.argv[1] == "M2":
        # Overlay Figure missingness cartoon
        overlayCartoon(
            fdir + "figureM2.svg",
            f"{cartoon_dir}/missingness_diagram.svg",
            75,
            5,
            scalee=1.1,
        )

    if sys.argv[1] == "M5":
        # Overlay Figure tumor vs NATs heatmap
        overlayCartoon(
            fdir + "figureM5.svg",
            f"{cartoon_dir}/heatmap_NATvsTumor.svg",
            50,
            0,
            scalee=0.40,
        )

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")
