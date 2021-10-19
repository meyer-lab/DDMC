"""
This creates Figure 2: Model figure
"""

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.colors as colors
import matplotlib.cm as cm
import logomaker as lm
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
from .common import subplotLabel, getSetup
from ..pre_processing import preprocessing, MeanCenter
from ..clustering import MassSpecClustering, KinToPhosphotypeDict
from ..binomial import AAlist
from ..plsr import R2Y_across_components
from .figure1 import import_phenotype_data, formatPhenotypesForModeling


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 6), (2, 4), multz={0: 1, 2: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import siganling data
    X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    # Fit DDMC
    ddmc = MassSpecClustering(i, ncl=5, SeqWeight=2, distance_method="PAM250").fit(d)
    centers = ddmc.transform()

    # Import phenotypes
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")
    c = import_phenotype_data(phenotype="Island")
    y = formatPhenotypesForModeling(cv, red, sw, c)
    y = y[y["Treatment"] == "A/E"].drop("Treatment", axis=1).set_index("Lines")

    # Pipeline diagram
    ax[0].axis("off")

    # Mass spec clustermap
    ax[1].axis("off")

    # Predictions
    Xs, models = ComputeCenters(X, d, i, ddmc, 5)
    Xs.append(centers)
    models.append("DDMC mix")
    plotStripActualVsPred(ax[2], [3, 4, 2, 3, 4], Xs, y, models)

    # Scores & Loadings
    lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]
    plsr = PLSRegression(n_components=4)
    plotScoresLoadings(ax[3:5], plsr.fit(centers, y), centers, y, ddmc.n_components, lines, pcX=1, pcY=2)

    # Plot upstream kinases heatmap
    plotDistanceToUpstreamKinase(ddmc, [1, 2, 3, 4, 5], ax[5], num_hits=10)

    return f


def plotGridSearch(ax, gs):
    """ Plot gridsearch results by ranking. """
    ax = sns.barplot(x="rank_test_score", y="mean_test_score", data=np.abs(gs.iloc[:20, :]), ax=ax, **{"linewidth": 0.5}, **{"edgecolor": "black"})
    ax.set_title("Hyperaparameter Search")
    ax.set_xticklabels(np.arange(1, 21))
    ax.set_ylabel("Mean Squared Error")


def plotR2YQ2Y(ax, model, X, Y, b=3, color="darkblue", title=False):
    """ Plot R2Y/Q2Y variance explained by each component. """
    Q2Y = R2Y_across_components(model, X, Y, b, crossval=True)
    R2Y = R2Y_across_components(model, X, Y, b)

    range_ = np.arange(1, b)

    ax.bar(range_ + 0.15, Q2Y, width=0.3, align="center", label="Q2Y", color=color, **{"linewidth": 0.5}, **{"edgecolor": "black"})
    ax.bar(range_ - 0.15, R2Y, width=0.3, align="center", label="R2Y", color="black", **{"linewidth": 0.5}, **{"edgecolor": "black"})
    ax.set_xticks(range_)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Variance")
    ax.legend(loc=0)
    if title:
        ax.set_title(title)


def ComputeCenters(X, d, i, ddmc, ncl):
    """Calculate cluster centers of  different algorithms."""
    # k-means
    labels = KMeans(n_clusters=ncl).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T

    # GMM
    ddmc_data = MassSpecClustering(i, ncl=ncl, SeqWeight=0, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    c_gmm = ddmc_data.transform()

    # DDMC seq
    ddmc_seq = MassSpecClustering(i, ncl=ncl, SeqWeight=ddmc.SeqWeight + 20, distance_method=ddmc.distance_method, random_state=ddmc.random_state).fit(d)
    ddmc_seq_c = ddmc_seq.transform()

    # DDMC mix
    ddmc_c = ddmc.transform()
    return [c_kmeans, c_gmm, ddmc_seq_c, ddmc_c], ["Unclustered", "k-means", "GMM", "DDMC seq", "DDMC mix"]


def plotStripActualVsPred(ax, n_components, Xs, Y, models):
    """Actual vs Predicted of different PLSR models"""
    datas = []
    for ii, X in enumerate(Xs):
        data = pd.DataFrame()
        Y_predictions = cross_val_predict(PLSRegression(n_components=n_components[ii]), X, Y, cv=Y.shape[0])
        coeff = [sp.stats.pearsonr(Y_predictions[:, jj], Y.iloc[:, jj])[0] for jj in range(len(Y.columns))]
        data["Phenotype"] = list(Y.columns)
        data["r-score"] = coeff
        data["Model"] = models[ii]
        datas.append(data)
    res = pd.concat(datas)
    sns.stripplot(x="Phenotype", y="r-score", data=res, ax=ax, hue="Model")
    ax.set_title("Actual vs Predicted")
    ax.legend(prop={'size': 8})


def plotActualVsPredicted(ax, plsr_model, X, Y, y_pred="cross-validation", color="darkblue", type="scatter", title=False):
    """ Plot exprimentally-measured vs PLSR-predicted values. """
    if y_pred == "cross-validation":
        Y_predictions = cross_val_predict(plsr_model, X, Y, cv=Y.shape[0])
        ylabel = "Predicted"
    if y_pred == "fit":
        Y_predictions = plsr_model.fit(X, Y).predict(X)
        ylabel = "Fit"

    if len(Y.columns) > 1:
        if type == "scatter":
            for i, label in enumerate(Y.columns):
                y = Y.iloc[:, i]
                ypred = Y_predictions[:, i]
                ax[i].scatter(y, ypred, color=color)
                ax[i].plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
                ax[i].set_xlabel("Actual")
                ax[i].set_ylabel(ylabel)
                ax[i].set_title(label)
                ax[i].set_aspect("equal", "datalim")
                add_rBox(ypred, y, ax)

        elif type == "bar":
            coeff = [sp.stats.pearsonr(Y_predictions[:, i], Y.iloc[:, i])[0] for i in range(len(Y.columns))]
            data = pd.DataFrame()
            data["Phenotype"] = list(Y.columns)
            data["r-score"] = coeff
            sns.barplot(x="Phenotype", y="r-score", data=data, ax=ax, color=color, **{"linewidth": 0.5}, **{"edgecolor": "black"})
            if title:
                ax.set_title(title)

    elif len(Y.columns) == 1:
        y = Y.iloc[:, 0]
        ypred = Y_predictions[:, 0]
        ax.scatter(y, ypred)
        ax.plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
        ax.set_xlabel("Actual")
        ax.set_ylabel(ylabel)
        ax.set_title(Y.columns[0])
        add_rBox(ypred, y, ax)


def add_rBox(ypred, y, ax):
    """Add correlation coefficient box onto scatter plot of Actual vs Predicted."""
    coeff, _ = sp.stats.pearsonr(ypred, y)
    textstr = "$r$ = " + str(np.round(coeff, 4))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax.text(0.75, 0.10, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, pcX=1, pcY=2, data="clusters", annotate=True, spacer=0.05):
    """Plot Scores and Loadings of PLSR model"""
    X_scores, _ = model.transform(X, Y)
    PC1_xload, PC2_xload = model.x_loadings_[:, pcX - 1], model.x_loadings_[:, pcY - 1]
    PC1_yload, PC2_yload = model.y_loadings_[:, pcX - 1], model.y_loadings_[:, pcY - 1]

    PC1_scores, PC2_scores = X_scores[:, pcX - 1], X_scores[:, pcY - 1]

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores)
    for j, txt in enumerate(treatments):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]), fontsize=10)
    ax[0].set_title("PLSR Model Scores", fontsize=12)
    ax[0].set_xlabel("Principal Component 1", fontsize=11)
    ax[0].set_ylabel("Principal Component 2", fontsize=11)
    ax[0].axhline(y=0, color="0.25", linestyle="--")
    ax[0].axvline(x=0, color="0.25", linestyle="--")

    ax[0].set_xlim([(-1 * max(np.abs(PC1_scores))) - spacer, max(np.abs(PC1_scores)) + spacer])
    ax[0].set_ylim([(-1 * max(np.abs(PC2_scores))) - spacer, max(np.abs(PC2_scores)) + spacer])

    # Loadings
    if data != "clusters":
        ncl = X.shape[1]

    colors_ = cm.rainbow(np.linspace(0, 1, ncl))
    if annotate:
        numbered = []
        list(map(lambda v: numbered.append(str(v + 1)), range(ncl)))
        for i, txt in enumerate(numbered):
            ax[1].annotate(txt, (PC1_xload[i] + spacer, PC2_xload[i] - spacer), fontsize=10)
    markers = ["x", "D", "*", "1"]
    for i, label in enumerate(Y.columns):
        ax[1].annotate(label, (PC1_yload[i] + spacer, PC2_yload[i] - spacer), fontsize=10)
        ax[1].scatter(PC1_yload[i], PC2_yload[i], color="black", marker=markers[i])
    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].set_title("PLSR Model Loadings", fontsize=12)
    ax[1].set_xlabel("Principal Component 1", fontsize=11)
    ax[1].set_ylabel("Principal Component 2", fontsize=11)
    ax[1].axhline(y=0, color="0.25", linestyle="--")
    ax[1].axvline(x=0, color="0.25", linestyle="--")


def plotCenters(ax, model, xlabels, yaxis=False, drop=False):
    centers = pd.DataFrame(model.transform()).T
    centers.columns = xlabels
    if drop:
        centers = centers.drop(drop)
    num_peptides = [np.count_nonzero(model.labels() == jj) for jj in range(1, model.n_components + 1)]
    for i in range(centers.shape[0]):
        cl = pd.DataFrame(centers.iloc[i, :]).T
        m = pd.melt(cl, value_vars=list(cl.columns), value_name="p-signal", var_name="Lines")
        m["p-signal"] = m["p-signal"].astype("float64")
        sns.lineplot(x="Lines", y="p-signal", data=m, color="#658cbb", ax=ax[i], linewidth=2)
        ax[i].set_xticklabels(xlabels, rotation=45)
        ax[i].set_xticks(np.arange(len(xlabels)))
        ax[i].set_ylabel("$log_{10}$ p-signal")
        ax[i].xaxis.set_tick_params(bottom=True)
        ax[i].set_xlabel("")
        ax[i].set_title("Cluster " + str(centers.index[i] + 1) + " Center " + "(" + "n=" + str(num_peptides[i]) + ")")
        if yaxis:
            ax[i].set_ylim([yaxis[0], yaxis[1]])


def plotMotifs(pssms, axes, titles=False, yaxis=False):
    """Generate logo plots of a list of PSSMs"""
    for i, ax in enumerate(axes):
        pssm = pssms[i].T
        if pssm.shape[0] == 11:
            pssm.index = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        elif pssm.shape[0] == 9:
            pssm.index = [-5, -4, -3, -2, -1, 1, 2, 3, 4]
        logo = lm.Logo(pssm,
                       font_name='Arial',
                       vpad=0.1,
                       width=.8,
                       flip_below=False,
                       center_values=False,
                       ax=ax)
        logo.ax.set_ylabel('information (bits)')
        logo.style_xticks(anchor=1, spacing=1)
        if titles:
            logo.ax.set_title(titles[i] + " Motif")
        else:
            logo.ax.set_title('Motif Cluster ' + str(i + 1))
        if yaxis:
            logo.ax.set_ylim([yaxis[0], yaxis[1]])


def plot_LassoCoef(ax, model, title=False):
    """Plot Lasso Coefficients"""
    coefs = pd.DataFrame(model.coef_).T
    coefs.index += 1
    coefs = coefs.reset_index()
    coefs.columns = ["Cluster", 'Viability', 'Apoptosis', 'Migration', 'Island']
    m = pd.melt(coefs, id_vars="Cluster", value_vars=list(coefs.columns)[1:], var_name="Phenotype", value_name="Coefficient")
    sns.barplot(x="Cluster", y="Coefficient", hue="Phenotype", data=m, ax=ax)
    if title:
        ax.set_title(title)


def store_cluster_members(X, model, filename, cols):
    """Save csv files with cluster members."""
    X["Cluster"] = model.labels()
    for i in range(model.n_components):
        m = X[X["Cluster"] == i + 1][cols]
        m.index = np.arange(m.shape[0])
        m.to_csv("msresist/data/cluster_members/" + filename + str(i + 1) + ".csv")


def plotDistanceToUpstreamKinase(model, clusters, ax, kind="strip", num_hits=5, additional_pssms=False, add_labels=False, title=False):
    """Plot Frobenius norm between kinase PSPL and cluster PSSMs"""
    ukin = model.predict_UpstreamKinases(additional_pssms=additional_pssms, add_labels=add_labels)
    ukin_mc = MeanCenter(ukin, mc_col=True, mc_row=True)
    if isinstance(add_labels, list):
        clusters += add_labels
    data = ukin_mc.sort_values(by="Kinase").set_index("Kinase")[clusters]
    if kind == "heatmap":
        sns.heatmap(data.T, ax=ax, xticklabels=data.index)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
        ax.set_ylabel("Cluster")

    elif kind == "strip":
        data = pd.melt(data.reset_index(), id_vars="Kinase", value_vars=list(data.columns), var_name="Cluster", value_name="Frobenius Distance")
        if isinstance(add_labels, list):
            # Actual ERK predictions
            data["Cluster"] = data["Cluster"].astype(str)
            d1 = data[~data["Cluster"].str.contains("_S")]
            sns.stripplot(data=d1, x="Cluster", y="Frobenius Distance", ax=ax[0])
            AnnotateUpstreamKinases(model, [21, 24, 27, "ERK2+"], ax[0], d1, 1)

            # Shuffled
            d2 = data[data["Kinase"] == "ERK2"]
            d2["Shuffled"] = ["_S" in s for s in d2["Cluster"]]
            d2["Cluster"] = [s.split("_S")[0] for s in d2["Cluster"]]
            sns.stripplot(data=d2, x="Cluster", y="Frobenius Distance", hue="Shuffled", ax=ax[1], size=8)
            ax[1].set_title("ERK2 Shuffled Positions")
            ax[1].legend(prop={'size': 10}, loc='lower left')
            DrawArrows(ax[1], d2)

        else:
            sns.stripplot(data=data, x="Cluster", y="Frobenius Distance", ax=ax)
            AnnotateUpstreamKinases(model, clusters, ax, data, num_hits)
            if title:
                ax.set_title(title)


def AnnotateUpstreamKinases(model, clusters, ax, data, num_hits=1):
    """Annotate upstream kinase predictions"""
    data.iloc[:, 1] = data.iloc[:, 1].astype(str)
    pssms, _ = model.pssms()
    for ii, c in enumerate(clusters, start=1):
        cluster = data[data.iloc[:, 1] == str(c)]
        hits = cluster.sort_values(by="Frobenius Distance", ascending=True)
        hits.index = np.arange(hits.shape[0])
        hits["Phosphoacceptor"] = [KinToPhosphotypeDict[kin] for kin in hits["Kinase"]]
        try:
            cCP = pssms[c - 1].iloc[:, 5].idxmax()
        except:
            cCP == "S/T"
        if cCP == "S" or cCP == "T":
            cCP = "S/T"
        hits = hits[hits["Phosphoacceptor"] == cCP]
        for jj in range(num_hits):
            ax.annotate(hits["Kinase"].iloc[jj], (ii - 1, hits["Frobenius Distance"].iloc[jj] - 0.01), fontsize=8)
    ax.legend().remove()
    ax.set_title("Kinase vs Cluster Motif")


def DrawArrows(ax, d2):
    data_shuff = d2[d2["Shuffled"]]
    actual_erks = d2[d2["Shuffled"] == False]
    arrow_lengths = np.add(data_shuff["Frobenius Distance"].values, abs(actual_erks["Frobenius Distance"].values)) * -1
    for dp in range(data_shuff.shape[0]):
        ax.arrow(dp,
                 data_shuff["Frobenius Distance"].iloc[dp] - 0.1,
                 0,
                 arrow_lengths[dp] + 0.3,
                 head_width=0.25,
                 head_length=0.15,
                 width=0.025,
                 fc='black',
                 ec='black')


def ShuffleClusters(shuffle, model, additional=False):
    """Returns PSSMs with shuffled positions"""
    ClustersToShuffle = np.array(shuffle)
    pssms, _ = model.pssms(PsP_background=False)
    s_pssms = []
    for s in ClustersToShuffle:
        mat = ShufflePositions(pssms[s])
        s_pssms.append(mat)

    if not isinstance(additional, bool):
        mat = ShufflePositions(additional)
        s_pssms.append(mat)

    return s_pssms


def ShufflePositions(pssm):
    """Shuffles the positions of input PSSMs"""
    pssm = np.array(pssm)
    mat = pssm[:, np.random.permutation([0, 1, 2, 3, 4, 6, 7, 8, 9])]
    mat = np.insert(mat, 5, pssm[:, 5], axis=1)
    mat = np.insert(mat, 1, pssm[:, -1], axis=1)
    mat = pd.DataFrame(mat)
    mat.index = AAlist
    return mat


def label_point(X, model, clusters, pspl, ax, n_neighbors=5):
    """Add labels to data points. Note not in use at the moment but could be helpful
    in the future (e.g. PCA of mass spec in figure 2)"""
    if isinstance(clusters, int):
        clusters = [clusters]
    pspl_ = pspl.copy()
    X_ = X.copy()
    for cluster in clusters:
        pssm = pd.DataFrame(X_.loc[cluster]).T.reset_index()
        pssm.columns = ["Label"] + list(pssm.columns[1:])
        XX = pd.concat([pspl_.reset_index(), pssm]).set_index("Label")
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(XX.values)
        idc = knn.kneighbors(XX.loc[cluster].values.reshape(1, 2), return_distance=False)
        a = XX.iloc[idc.reshape(n_neighbors), :].reset_index()
        a.columns = ["val", "x", "y"]
        for _, point in a.iterrows():
            ax.text(point['x'] + .02, point['y'], str(point['val']))
