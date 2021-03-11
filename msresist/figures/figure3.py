"""
This creates Figure 2.
"""
import os
import pickle
import pandas as pd
import numpy as np
import scipy as sp
from .common import subplotLabel, getSetup
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from ..plsr import R2Y_across_components
from ..figures.figure1 import pca_dfs
from ..distances import DataFrameRipleysK
import matplotlib.colors as colors
from sklearn.model_selection import cross_val_predict
import matplotlib.cm as cm
import seaborn as sns
import logomaker as lm
from ..pre_processing import preprocessing, y_pre, FixColumnLabels, MeanCenter
import warnings
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)

path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((17, 20), (4, 3), multz={10: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # -------- Import and Preprocess Signaling Data -------- #
    X = preprocessing(Axlmuts_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)

    d = X.select_dtypes(include=["float64"]).T

    all_lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    d.index = all_lines

    # -------- Cell Phenotypes -------- #
    # Cell Viability
    cv1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR1_Phase.csv")
    cv2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR2_Phase.csv")
    cv3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv")
    cv4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv")

    itp = 24
    ftp = 96

    cv = [cv1, cv2, cv3, cv4]
    cv = FixColumnLabels(cv)

    v_ut = y_pre(cv, "UT", ftp, "Viability", all_lines, itp=itp)
    v_e = y_pre(cv, "-E", ftp, "Viability", all_lines, itp=itp)
    v_ae = y_pre(cv, "A/E", ftp, "Viability", all_lines, itp=itp)

    # Cell Death
    red1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR1_RedCount.csv")
    red2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR2_RedCount.csv")
    red3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR3_RedCount.csv")
    red4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Red/BR4_RedCount.csv")
    red4.columns = red3.columns

    for jj in range(1, red1.columns.size):
        red1.iloc[:, jj] /= cv1.iloc[:, jj]
        red2.iloc[:, jj] /= cv2.iloc[:, jj]
        red3.iloc[:, jj] /= cv3.iloc[:, jj]
        red4.iloc[:, jj] /= cv4.iloc[:, jj]

    cD = [red1, red2, red3, red4]
    cD = FixColumnLabels(cD)
    cd_ut = y_pre(cD, "UT", ftp, "Apoptosis", all_lines, itp=itp)
    cd_e = y_pre(cD, "-E", ftp, "Apoptosis", all_lines, itp=itp)
    cd_ae = y_pre(cD, "A/E", ftp, "Apoptosis", all_lines, itp=itp)

    r1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR1_RWD.csv")
    r2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR2_RWD.csv")
    r3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR3_RWD.csv")
    r4 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/BR4_RWD.csv")
    ftp = 12
    cm = [r1, r2, r3, r4]
    m_ut = y_pre(cm, "UT", ftp, "Migration", all_lines)
    m_e = y_pre(cm, " E", ftp, "Migration", all_lines)
    m_ae = y_pre(cm, "A/E", ftp, "Migration", all_lines)

    m_ut.index = v_ut.index
    m_e.index = v_e.index
    m_ae.index = v_ae.index

    # Clustering Effect
    mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
    treatments = ['ut', 'e', 'ae']
    replicates = 6
    radius = np.linspace(1, 14.67, 1)
    folder = '48hrs'
    c = DataFrameRipleysK(folder, mutants, treatments, replicates, radius).reset_index().set_index("Mutant")
    c.columns = ["Treatment", "Island"]
    c_ut = c[c["Treatment"] == "ut"]
    c_ut = c_ut.reindex(list(mutants[:2]) + [mutants[3]] + [mutants[2]] + list(mutants[4:]))
    c_ut.index = all_lines
    c_ut = c_ut.reset_index()
    c_ut["Treatment"] = "UT"

    c_e = c[c["Treatment"] == "e"]
    c_e = c_e.reindex(list(mutants[:2]) + [mutants[3]] + [mutants[2]] + list(mutants[4:]))
    c_e.index = all_lines
    c_e = c_e.reset_index()
    c_e["Treatment"] = "E"

    c_ae = c[c["Treatment"] == "ae"]
    c_ae = c_ae.reindex(list(mutants[:2]) + [mutants[3]] + [mutants[2]] + list(mutants[4:]))
    c_ae.index = all_lines
    c_ae = c_ae.reset_index()
    c_ae["Treatment"] = "A/E"

    # -------- PLOTS -------- #
    # PCA analysis of phenotypes
    y_ae = pd.concat([v_ae, cd_ae["Apoptosis"], m_ae["Migration"], c_ae["Island"]], axis=1)
    y_e = pd.concat([v_e, cd_e["Apoptosis"], m_e["Migration"], c_e["Island"]], axis=1)
    y_ut = pd.concat([v_ut, cd_ut["Apoptosis"], m_ut["Migration"], c_ut["Island"]], axis=1)

    y_c = pd.concat([y_ut, y_e, y_ae])
    y_c.iloc[:, 2:] = StandardScaler().fit_transform(y_c.iloc[:, 2:])

    plotPCA(ax[:2], y_c, 2, ["Lines", "Treatment"], "Phenotype", hue_scores="Lines", style_scores="Treatment", legendOut=True)

    # MODEL
    y = y_ae.drop("Treatment", axis=1).set_index("Lines")

    # -------- Cross-validation 1 -------- #
    # R2Y/Q2Y

    with open('msresist/data/pickled_models/AXLmodel_PAM250_W2_5CL', 'rb') as m:
        model = pickle.load(m)[0]
    centers = model.transform()

    plsr = PLSRegression(n_components=2, scale=False)
    plotR2YQ2Y(ax[2], plsr, centers, y, model.ncl + 1)

    # Plot Measured vs Predicted
    plotActualVsPredicted(ax[3:7], plsr, centers, y)

    # Plot motifs
    pssms = model.pssms(PsP_background=True)
    plotMotifs([pssms[0], pssms[3], pssms[4]], axes=ax[7:10], titles=["Cluster 1", "Cluster 4", "Cluster 5"])

    # Plot upstream kinases heatmap
    plotUpstreamKinase_heatmap(model, [1, 4, 5], ax[10])

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotPCA(ax, d, n_components, scores_ind, loadings_ind, hue_scores=None, style_scores=None, pvals=None, style_load=None, legendOut=False):
    """ Plot PCA scores and loadings. """
    pp = PCA(n_components=n_components)
    dScor_ = pp.fit_transform(d.select_dtypes(include=["float64"]).values)
    dLoad_ = pp.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, d, n_components, scores_ind, loadings_ind)
    varExp = np.round(pp.explained_variance_ratio_, 2)

    # Scores
    sns.scatterplot(x="PC1", y="PC2", data=dScor_, hue=hue_scores, style=style_scores, ax=ax[0], **{"linewidth": 0.5, "edgecolor": "k"})
    ax[0].set_title("PCA Scores")
    ax[0].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[0].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    if legendOut:
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2)

    # Loadings
    if isinstance(pvals, np.ndarray):
        dLoad_["p-value"] = pvals
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue="p-value", style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})
    else:
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})

    ax[1].set_title("PCA Loadings")
    ax[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    for j, txt in enumerate(dLoad_[loadings_ind]):
        ax[1].annotate(txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10)


def plotGridSearch(ax, gs):
    """ Plot gridsearch results by ranking. """
    ax = sns.barplot(x="rank_test_score", y="mean_test_score", data=np.abs(gs.iloc[:20, :]), ax=ax, **{"linewidth": 0.5}, **{"edgecolor": "black"})
    ax.set_title("Hyperaparameter Search")
    ax.set_xticklabels(np.arange(1, 21))
    ax.set_ylabel("Mean Squared Error")


def plotR2YQ2Y(ax, model, X, Y, b=3, color="darkblue"):
    """ Plot R2Y/Q2Y variance explained by each component. """
    Q2Y = R2Y_across_components(model, X, Y, b, crossval=True)
    R2Y = R2Y_across_components(model, X, Y, b)

    range_ = np.arange(1, b)

    ax.bar(range_ + 0.15, Q2Y, width=0.3, align="center", label="Q2Y", color=color)
    ax.bar(range_ - 0.15, R2Y, width=0.3, align="center", label="R2Y", color="black")
    ax.set_title("R2Y/Q2Y - Cross-validation", fontsize=12)
    ax.set_xticks(range_)
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Variance", fontsize=11)
    ax.legend(loc=0)


def plotActualVsPredicted(ax, plsr_model, X, Y, y_pred="cross-validation", color="darkblue"):
    """ Plot exprimentally-measured vs PLSR-predicted values. """
    if y_pred == "cross-validation":
        Y_predictions = cross_val_predict(plsr_model, X, Y, cv=Y.shape[0])
        ylabel = "Predicted"
    if y_pred == "fit":
        Y_predictions = plsr_model.fit(X, Y).predict(X)
        ylabel = "Fit"

    if len(Y.columns) > 1:
        for i, label in enumerate(Y.columns):
            y = Y.iloc[:, i]
            ypred = Y_predictions[:, i]
            ax[i].scatter(y, ypred, color=color)
            ax[i].plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
            ax[i].set_xlabel("Actual")
            ax[i].set_ylabel(ylabel)
            ax[i].set_title(label)

            ax[i].set_aspect("equal", "datalim")

            # Add correlation coefficient
            coeff, _ = sp.stats.pearsonr(ypred, y)
            textstr = "$r$ = " + str(np.round(coeff, 4))
            props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
            ax[i].text(0.75, 0.10, textstr, transform=ax[i].transAxes, verticalalignment="top", bbox=props)

    elif len(Y.columns) == 1:
        y = Y.iloc[:, 0]
        ypred = Y_predictions[:, 0]
        ax.scatter(y, ypred)
        ax.plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
        ax.set_xlabel("Actual")
        ax.set_ylabel(ylabel)
        ax.set_title(Y.columns[0])

        # Add correlation coefficient
        coeff, _ = sp.stats.pearsonr(ypred, y)
        textstr = "$r$ = " + str(np.round(coeff, 4))
        props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
        ax.text(0.75, 0.10, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, pcX=1, pcY=2, data="clusters", annotate=True):
    """Plot Scores and Loadings of PLSR model"""
    X_scores, _ = model.transform(X, Y)
    PC1_xload, PC2_xload = model.x_loadings_[:, pcX - 1], model.x_loadings_[:, pcY - 1]
    PC1_yload, PC2_yload = model.y_loadings_[:, pcX - 1], model.y_loadings_[:, pcY - 1]

    PC1_scores, PC2_scores = X_scores[:, pcX - 1], X_scores[:, pcY - 1]

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores)
    for j, txt in enumerate(treatments):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    ax[0].set_title("PLSR Model Scores", fontsize=12)
    ax[0].set_xlabel("Principal Component 1", fontsize=11)
    ax[0].set_ylabel("Principal Component 2", fontsize=11)
    ax[0].axhline(y=0, color="0.25", linestyle="--")
    ax[0].axvline(x=0, color="0.25", linestyle="--")

    spacer = 0.5
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
            ax[1].annotate(txt, (PC1_xload[i], PC2_xload[i]))
    markers = ["x", "D", "*", "1"]
    for i, label in enumerate(Y.columns):
        ax[1].annotate(label, (PC1_yload[i] + 0.001, PC2_yload[i] - 0.001))
        ax[1].scatter(PC1_yload[i], PC2_yload[i], color="black", marker=markers[i])
    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].set_title("PLSR Model Loadings", fontsize=12)
    ax[1].set_xlabel("Principal Component 1", fontsize=11)
    ax[1].set_ylabel("Principal Component 2", fontsize=11)
    ax[1].axhline(y=0, color="0.25", linestyle="--")
    ax[1].axvline(x=0, color="0.25", linestyle="--")


def plotCenters(ax, centers, xlabels, title, yaxis=False):
    centers = pd.DataFrame(centers).T
    centers.columns = xlabels
    for i in range(centers.shape[0]):
        cl = pd.DataFrame(centers.iloc[i, :]).T
        m = pd.melt(cl, value_vars=list(cl.columns), value_name="p-signal", var_name="Lines")
        m["p-signal"] = m["p-signal"].astype("float64")
        sns.lineplot(x="Lines", y="p-signal", data=m, color="#658cbb", ax=ax, linewidth=2)
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_ylabel("$log_{10}$ p-signal")
        ax.xaxis.set_tick_params(bottom=True)
        ax.set_xlabel("")
        if title:
            ax.legend([title])
        else:
            ax.legend(["cluster " + str(i + 1)])
        if yaxis:
            ax.set_ylim([yaxis[0], yaxis[1]])


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
    for i in range(model.ncl):
        m = X[X["Cluster"] == i + 1][cols]
        m.index = np.arange(m.shape[0])
        m.to_csv("msresist/data/cluster_members/" + filename + str(i + 1) + ".csv")


def plotUpstreamKinase_heatmap(model, clusters, ax):
    """Plot Frobenius norm between kinase PSPL and cluster PSSMs"""
    ukin = model.predict_UpstreamKinases()
    ukin_mc = MeanCenter(ukin, mc_col=True, mc_row=True)
    ukin_mc.columns = ["Kinase"] + list(np.arange(1, model.ncl + 1))
    data = ukin_mc.sort_values(by="Kinase").set_index("Kinase")[clusters]
    sns.heatmap(data.T, ax=ax, xticklabels=data.index, cbar_kws={"shrink": 0.75})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    ax.set_ylabel("Frobenius Norm (motif vs kinase specifcity)")


def label_point(X, model, clusters, pspl, ax, n_neighbors=5):
    """Add labels to data points"""
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
