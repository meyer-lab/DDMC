"""Shared plotting and figure-assembly helpers used across `ddmc/figures/figureM*.py`.

Contains:
    - `getSetup` / `subplotLabel` / `overlayCartoon`: build a labeled
      multi-panel matplotlib figure and overlay static SVG cartoons onto it.
    - `genFigure`: the `fbuild` console-script entry point (see
      `pyproject.toml`) that generates a given `figureM*.py` module's
      figure and saves it to `./output/`.
    - `plot_motifs`: sequence-logo plot of a `DDMC` cluster's PSSM.
    - `plot_cluster_kinase_distances`: strip plot of kinase-vs-cluster PSSM
      distances, annotated with the top kinase hit(s) per cluster.
    - `get_pvals_across_clusters` / `plot_p_signal_across_clusters_and_binary_feature`:
      statistically compare cluster centers between two groups of samples
      and plot the result as an annotated violin plot.
    - `plot_pca_on_cluster_centers`: PCA scores/loadings plot of cluster
      centers.
"""

import importlib
import sys
import time
from collections.abc import Sequence
from string import ascii_uppercase

import logomaker as lm
import numpy as np
import pandas as pd
import seaborn as sns
import svgutils.transform as st
from matplotlib import axes, gridspec, rcParams
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

from ddmc.binomial import AAlist

from ..motifs import KinToPhosphotypeDict

rcParams["font.sans-serif"] = "Arial"


def getSetup(
    figsize: tuple[int, int],
    gridd: tuple[int, int],
    multz: None | dict = None,
    labels: bool = True,
) -> tuple:
    """Establish figure set-up with subplots.

    Args:
        figsize: Figure size in inches, as `(width, height)`.
        gridd: Subplot grid shape, as `(n_rows, n_cols)`.
        multz: Maps a subplot's flat grid index to how many extra
            consecutive grid cells it should span (e.g. `{0: 2}` makes the
            subplot at index 0 span indices 0-2). Spanned indices are
            skipped when placing subsequent subplots.
        labels: Whether to add bold uppercase letter labels (A, B, C, ...)
            to each subplot via `subplotLabel`.

    Returns:
        A tuple `(ax, f)` of the list of created `Axes` (in grid order) and
        the parent `Figure`.
    """
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
    gs1 = gridspec.GridSpec(gridd[0], gridd[1], figure=f)

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

    if labels:
        subplotLabel(ax)

    return (ax, f)


def subplotLabel(axs: list[axes.Axes]) -> None:
    """Place bold uppercase letter labels (A, B, C, ...) above each axes.

    Args:
        axs: Axes to label, in the order they should be lettered.
    """
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
    figFile: str, cartoonFile: str, x: float, y: float, scalee: float = 1.0
) -> None:
    """Overlay a static SVG cartoon onto a saved figure, in place.

    Args:
        figFile: Path to the SVG figure to overlay onto and overwrite.
        cartoonFile: Path to the SVG cartoon to overlay.
        x: X position (in SVG units) to place the cartoon's origin.
        y: Y position (in SVG units) to place the cartoon's origin.
        scalee: Uniform scale factor applied to the cartoon.
    """

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale_x=scalee, scale_y=scalee)

    template.append(cartoon)
    template.save(figFile)


def genFigure() -> None:
    """Console-script entry point (`fbuild`, see `pyproject.toml`) for
    generating one paper figure.

    Reads the figure name suffix from `sys.argv[1]` (e.g. `"M2"`), imports
    the corresponding `ddmc.figures.figureM2` module, calls its
    `makeFigure()`, and saves the result to `./output/figureM2.svg`. Some
    figures (`M2`, `M5`) additionally get a static SVG cartoon overlaid via
    `overlayCartoon` after saving.
    """
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    module = importlib.import_module(f"ddmc.figures.{nameOut}")
    ff = module.makeFigure()

    if ff is not None:
        ff.savefig(
            f"./output/{nameOut}.svg", dpi=300, bbox_inches="tight", pad_inches=0
        )

    if sys.argv[1] == "M2":
        # Overlay Figure missingness cartoon
        overlayCartoon(
            "./output/figureM2.svg",
            "./ddmc/figures/missingness_diagram.svg",
            75,
            5,
            scalee=1.1,
        )

    if sys.argv[1] == "M5":
        # Overlay Figure tumor vs NATs heatmap
        overlayCartoon(
            "./output/figureM5.svg",
            "./ddmc/figures/heatmap_NATvsTumor.svg",
            50,
            0,
            scalee=0.40,
        )

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def plot_motifs(pssm, ax: axes.Axes, titles=False, yaxis=False) -> None:
    """Draw a sequence-logo plot of a single cluster's PSSM.

    Args:
        pssm (numpy.ndarray | pandas.DataFrame): Position-specific scoring
            matrix of shape (20, 11) or (20, 9), e.g. one entry from
            `ddmc.clustering.DDMC.get_pssms`.
        ax: Axes to plot onto.
        titles (str | bool): If given (and not `False`), used as the axes
            title (with `" Motif"` appended); otherwise defaults to
            `"Motif Cluster 1"`.
        yaxis (Sequence[float] | bool): If given (and not `False`), a
            `[ymin, ymax]` pair to set the y-axis limits to.
    """
    pssm = pssm.T
    pssm = pd.DataFrame(pssm)
    if pssm.shape[0] == 11:
        pssm.index = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    elif pssm.shape[0] == 9:
        pssm.index = [-5, -4, -3, -2, -1, 1, 2, 3, 4]
    pssm.columns = AAlist
    logo = lm.Logo(
        pssm,
        font_name="Arial",
        vpad=0.1,
        width=0.8,
        flip_below=False,
        center_values=False,
        ax=ax,
    )
    ax.set_ylabel("log_{2} (Enrichment Score)")
    logo.style_xticks(anchor=1, spacing=1)
    if titles:
        ax.set_title(titles + " Motif")
    else:
        ax.set_title("Motif Cluster 1")
    if yaxis:
        ax.set_ylim(yaxis[0], yaxis[1])


def plot_cluster_kinase_distances(
    distances: pd.DataFrame, pssms: np.ndarray, ax: axes.Axes, num_hits: int = 1
) -> None:
    """Strip plot of kinase-vs-cluster PSSM distances, annotated with the
    top predicted kinase(s) per cluster.

    For each cluster, restricts candidate kinases to those whose known
    phosphoacceptor (`ddmc.motifs.KinToPhosphotypeDict`) matches that
    cluster's most frequent phosphoacceptor, then annotates the closest
    `num_hits` of those.

    Args:
        distances: Kinase-by-cluster Frobenius distance matrix (kinases as
            rows, cluster/PSSM names as columns), as returned by
            `ddmc.clustering.DDMC.predict_upstream_kinases`.
        pssms: The PSSMs corresponding to `distances`'s columns, of shape
            (n_clusters, 20, 11), used to determine each cluster's dominant
            phosphoacceptor.
        ax: Axes to plot onto.
        num_hits: Number of top kinase hits to annotate per cluster.
    """
    pssm_names = distances.columns

    # these centering lines make no sense, but they were used in the original
    # publication-version of this code
    distances = distances.sub(distances.mean(axis=1), axis=0)
    distances = distances.sub(distances.mean(axis=0), axis=1)

    # melt distances
    distances = pd.melt(
        distances.reset_index(names="Kinase"),
        id_vars="Kinase",
        value_vars=list(distances.columns),
        var_name="PSSM name",
        value_name="Frobenius Distance",
    )

    sns.stripplot(data=distances, x="PSSM name", y="Frobenius Distance", ax=ax)

    # Annotate upstream kinase predictions
    for i, pssm_name in enumerate(pssm_names):
        distances_pssm = distances[distances["PSSM name"] == pssm_name]
        distances_pssm = distances_pssm.sort_values(
            by="Frobenius Distance", ascending=True
        )
        distances_pssm = distances_pssm.reset_index(drop=True)
        # assert that the kinase phosphoacceptor and most frequent phosphoacceptor in the pssm match
        distances_pssm["Phosphoacceptor"] = [
            KinToPhosphotypeDict[kin] for kin in distances_pssm["Kinase"]
        ]
        try:
            most_frequent_phosphoacceptor = AAlist[int(np.argmax(pssms[i, :, 5]))]
        except Exception:
            most_frequent_phosphoacceptor = "S/T"
        if most_frequent_phosphoacceptor == "S" or most_frequent_phosphoacceptor == "T":
            most_frequent_phosphoacceptor = "S/T"
        distances_pssm = distances_pssm[
            distances_pssm["Phosphoacceptor"] == most_frequent_phosphoacceptor
        ]
        for jj in range(num_hits):
            ax.annotate(
                distances_pssm["Kinase"].iloc[jj],
                (i, distances_pssm["Frobenius Distance"].iloc[jj] - 0.01),
                fontsize=8,
            )
    ax.legend().remove()
    ax.set_title("Kinase vs Cluster Motif")


def get_pvals_across_clusters(
    label: pd.Series | np.ndarray, centers: pd.DataFrame | np.ndarray
) -> np.ndarray:
    """Test whether each cluster's center differs between two groups of samples.

    Runs a Mann-Whitney U test per cluster between the samples where
    `label` is True and where it's False, then corrects for multiple
    testing across clusters.

    Args:
        label: Boolean mask of shape (n_samples,) splitting samples into
            two groups (e.g. tumor vs. NAT).
        centers: Cluster centers of shape (n_samples, n_components),
            aligned to `label`.

    Returns:
        Multiple-testing-corrected p-value for each cluster, of shape
        (n_components,).
    """
    pvals = []
    if isinstance(centers, pd.DataFrame):
        centers = centers.values
    centers_pos = centers[label]
    centers_neg = centers[~label]
    for i in range(centers.shape[1]):
        pvals.append(mannwhitneyu(centers_pos[:, i], centers_neg[:, i])[1])
    return multipletests(pvals)[1]


def plot_p_signal_across_clusters_and_binary_feature(
    feature: pd.Series | np.ndarray,
    centers: pd.DataFrame,
    label_name: str,
    ax: axes.Axes,
) -> None:
    """Violin-plot cluster centers split by a binary sample feature, with a
    significance marker on each cluster whose center differs between the
    two groups (see `get_pvals_across_clusters`).

    Args:
        feature: Boolean mask of shape (n_samples,) splitting samples into
            two groups (e.g. tumor vs. NAT), aligned to `centers`'s rows.
        centers: Cluster centers of shape (n_samples, n_components), e.g.
            from `ddmc.clustering.DDMC.transform(as_df=True)`.
        label_name: Name to use for `feature` in the plot legend.
        ax: Axes to plot onto.
    """
    centers = centers.copy()
    centers_labeled = centers.copy()
    centers_labeled[label_name] = feature
    df_violin = centers_labeled.reset_index().melt(
        id_vars=label_name,
        value_vars=centers.columns,
        value_name="p-signal",
        var_name="Cluster",
    )
    sns.violinplot(
        data=df_violin,
        x="Cluster",
        y="p-signal",
        hue=label_name,
        dodge=True,
        ax=ax,
        linewidth=0.25,
    )
    ax.legend(prop={"size": 8})
    annotation_height = df_violin["p-signal"].max() + 0.02
    for i, pval in enumerate(get_pvals_across_clusters(feature, centers)):
        if pval < 0.05:
            annotation = "*"
        elif pval < 0.01:
            annotation = "**"
        else:
            continue
        ax.text(i, annotation_height, annotation, ha="center", va="bottom", fontsize=10)


def plot_pca_on_cluster_centers(
    centers: pd.DataFrame,
    axes: Sequence,
    hue_scores: Sequence | np.ndarray | None = None,
    hue_scores_title: str | None = None,
    hue_loadings: Sequence | np.ndarray | None = None,
    hue_loadings_title: str | None = None,
) -> None:
    """Plot a 2-component PCA of cluster centers, as a scores plot (one
    point per sample) and a loadings plot (one point per cluster).

    Args:
        centers: Cluster centers of shape (n_samples, n_components), e.g.
            from `ddmc.clustering.DDMC.transform(as_df=True)`.
        axes: A length-2 sequence of Axes: `axes[0]` for the scores plot,
            `axes[1]` for the loadings plot.
        hue_scores: Per-sample values to color the scores plot points by.
        hue_scores_title: If given, shown as the scores plot's legend title.
        hue_loadings: Per-cluster values to color the loadings plot points
            by.
        hue_loadings_title: If given, adds a `"p < 0.01"`-titled legend to
            the loadings plot (its entries come from `hue_loadings`, not
            this string).
    """
    # run PCA on cluster centers
    pca = PCA(n_components=2)
    scores = pca.fit_transform(centers)  # sample by PCA component
    loadings = pca.components_  # PCA component by cluster
    variance_explained = np.round(pca.explained_variance_ratio_, 2)

    # plot scores
    sns.scatterplot(
        x=scores[:, 0],
        y=scores[:, 1],
        hue=hue_scores,
        ax=axes[0],
        **{"linewidth": 0.5, "edgecolor": "k"},
    )
    if hue_scores_title:
        axes[0].legend(
            loc="lower left", prop={"size": 9}, title=hue_scores_title, fontsize=9
        )
    axes[0].set_title("PCA Scores")
    axes[0].set_xlabel(
        "PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10
    )
    axes[0].set_ylabel(
        "PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10
    )

    # plot loadings
    sns.scatterplot(
        x=loadings[0],
        y=loadings[1],
        ax=axes[1],
        hue=hue_loadings,
        **{"linewidth": 0.5, "edgecolor": "k"},
    )
    if hue_loadings_title:
        axes[1].legend(title="p < 0.01", prop={"size": 8})
    axes[1].set_title("PCA Loadings")
    axes[1].set_xlabel(
        "PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10
    )
    axes[1].set_ylabel(
        "PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10
    )
    for j, txt in enumerate(centers.columns):
        axes[1].annotate(
            txt, (loadings[0][j] + 0.001, loadings[1][j] + 0.001), fontsize=10
        )
