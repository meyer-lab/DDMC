""" PCA functions """

import pandas as pd
import numpy as np
import seaborn as sns
from msresist.pre_processing import Linear
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import explained_variance_score, r2_score


def pca_dfs(scores, loadings, df, n_components, sIDX, lIDX):
    """ build PCA scores and loadings data frames. """
    dScor = pd.DataFrame()
    dLoad = pd.DataFrame()
    for i in range(n_components):
        cpca = "PC" + str(i + 1)
        dScor[cpca] = scores[:, i]
        dLoad[cpca] = loadings[i, :]

    for j in sIDX:
        dScor[j] = list(df[j])
    dLoad[lIDX] = df.select_dtypes(include=["float64"]).columns
    return dScor, dLoad


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
    ax[0].legend(prop={'size': 8})
    if legendOut:
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2, prop={'size': 8})

    # Loadings
    if isinstance(pvals, np.ndarray):
        dLoad_["p-value"] = pvals
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue="p-value", style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})
    else:
        sns.scatterplot(x="PC1", y="PC2", data=dLoad_, style=style_load, ax=ax[1], **{"linewidth": 0.5, "edgecolor": "k"})

    ax[1].set_title("PCA Loadings")
    ax[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    ax[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    ax[1].legend(prop={'size': 8})
    for j, txt in enumerate(dLoad_[loadings_ind]):
        ax[1].annotate(txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10)


def plotPCA_scoresORloadings(ax, d, n_components, scores_ind, loadings_ind, hue_scores=None, style_scores=None, pvals=None, style_load=None, legendOut=False, plot="scores", annotateScores=False):
    """Plot PCA scores only"""
    pp = PCA(n_components=n_components)
    dScor_ = pp.fit_transform(d.select_dtypes(include=["float64"]).values)
    dLoad_ = pp.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, d, n_components, scores_ind, loadings_ind)
    varExp = np.round(pp.explained_variance_ratio_, 2)

    # Scores
    if plot == "scores":
        sns.scatterplot(x="PC1", y="PC2", data=dScor_, hue=hue_scores, style=style_scores, ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})
        ax.set_title("PCA Scores")
        ax.set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
        ax.set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
        if legendOut:
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, labelspacing=0.2, prop={'size': 8})
        if annotateScores:
            for j, txt in enumerate(d[scores_ind[0]]):
                ax.annotate(txt, (dScor_["PC1"][j] + 0.001, dScor_["PC2"][j] + 0.001), fontsize=10)

    # Loadings
    elif plot == "loadings":
        if isinstance(pvals, np.ndarray):
            dLoad_["p-value"] = pvals
            sns.scatterplot(x="PC1", y="PC2", data=dLoad_, hue="p-value", style=style_load, ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})
        else:
            sns.scatterplot(x="PC1", y="PC2", data=dLoad_, style=style_load, ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})

        ax.set_title("PCA Loadings")
        ax.set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
        ax.set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
        for j, txt in enumerate(dLoad_[loadings_ind]):
            ax.annotate(txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10)


def plotpca_explained(ax, data, ncomp):
    """ Cumulative variance explained for each principal component. """
    explained = PCA(n_components=ncomp).fit(data).explained_variance_ratio_
    acc_expl = []

    for i, exp in enumerate(explained):
        if i > 0:
            exp += acc_expl[i - 1]
            acc_expl.append(exp)
        else:
            acc_expl.append(exp)

    ax.bar(range(ncomp), acc_expl, edgecolor="black", linewidth=1)
    ax.set_ylabel("% Variance Explained")
    ax.set_xlabel("Components")
    ax.set_xticks(range(ncomp))
    ax.set_xticklabels([i + 1 for i in range(ncomp)])


def plotpca_ScoresLoadings(ax, data, pn, ps):
    fit = PCA(n_components=2).fit(data)
    PC1_scores, PC2_scores = fit.transform(data)[:, 0], fit.transform(data)[:, 1]
    PC1_loadings, PC2_loadings = fit.components_[0], fit.components_[1]

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores, linewidths=0.2)
    for j, txt in enumerate(list(data.index)):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    ax[0].set_title("PCA Scores")
    ax[0].set_xlabel("Principal Component 1")
    ax[0].set_ylabel("Principal Component 2")
    ax[0].axhline(y=0, color="0.25", linestyle="--")
    ax[0].axvline(x=0, color="0.25", linestyle="--")

    spacer = 0.5
    ax[0].set_xlim([(-1 * max(np.abs(PC1_scores))) - spacer, max(np.abs(PC1_scores)) + spacer])
    ax[0].set_ylim([(-1 * max(np.abs(PC2_scores))) - spacer, max(np.abs(PC2_scores)) + spacer])

    # Loadings
    poi = [
        "AXL Y702-p",
        "AXL Y866-p",
        "AXL Y481-p",
        "RAB13 Y5-p",
        "RAB2B Y3-p",
        "CBLB Y889-p",
        "SOS1 Y1196-p",
        "GAB2 T265-p",
        "GAB1 Y406",
        "CRKL Y251-p",
        "PACSIN2 Y388-p",
        "SHC1 S426-p",
        "GSK3A Y279-p",
        "NME2 Y52-p",
        "CDK1 Y15-p",
        "MAPK3 T207-p",
        "TNS3 Y802-p",
        "GIT1 Y383-p",
        "KIRREL1 Y622-p",
        "BCAR1 Y234-p",
        "NEDD9 S182-p",
        "RIN1 Y681-p",
        "ATP8B1 Y1217",
        "MAPK3 Y204-p",
        "ATP1A1 Y55-p",
        "YES1 Y446",
        "EPHB3 Y792-p",
        "SLC35E1 Y403-p",
    ]
    for i, name in enumerate(pn):
        p = name + " " + ps[i]
        if p in poi:
            ax[1].annotate(name, (PC1_loadings[i], PC2_loadings[i]))
    ax[1].scatter(PC1_loadings, PC2_loadings, c="darkred", linewidths=0.2, alpha=0.7)
    ax[1].set_title("PCA Loadings")
    ax[1].set_xlabel("Principal Component 1")
    ax[1].set_ylabel("Principal Component 2")
    ax[1].axhline(y=0, color="0.25", linestyle="--")
    ax[1].axvline(x=0, color="0.25", linestyle="--")

    spacer = 0.05
    ax[1].set_xlim([(-1 * max(np.abs(PC1_loadings)) - spacer), (max(np.abs(PC1_loadings)) + spacer)])
    ax[1].set_ylim([(-1 * max(np.abs(PC2_loadings)) - spacer), (max(np.abs(PC2_loadings)) + spacer)])


def plotpca_ScoresLoadings_plotly(data, title, loc=False):
    """ Interactive PCA plot. Note that this works best by pre-defining the dataframe's
    indices which will serve as labels for each dot in the plot. """
    fit = PCA(n_components=2).fit(data)

    scores = pd.concat([pd.DataFrame(fit.transform(data)[:, 0]), pd.DataFrame(fit.transform(data)[:, 1])], axis=1)
    scores.index = data.index
    scores.columns = ["PC1", "PC2"]

    loadings = pd.concat([pd.DataFrame(fit.components_[0]), pd.DataFrame(fit.components_[1])], axis=1)
    loadings.index = data.columns
    loadings.columns = ["PC1", "PC2"]

    if loc:
        print(loadings.loc[loc])

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA Scores", "PCA Loadings"))
    fig.add_trace(
        go.Scatter(
            mode="markers+text",
            x=scores["PC1"],
            y=scores["PC2"],
            text=scores.index,
            textposition="top center",
            textfont=dict(size=10, color="black"),
            marker=dict(color="blue", size=8, line=dict(color="black", width=1)),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=loadings["PC1"],
            y=loadings["PC2"],
            opacity=0.7,
            text=["Protein: " + loadings.index[i][0] + "  Pos: " + loadings.index[i][1] for i in range(len(loadings.index))],
            marker=dict(color="crimson", size=8, line=dict(color="black", width=1)),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500,
        width=1000,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False),
        title_text=title,
    ),
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=1)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)

    fig.show()


def preprocess_ID(linear=False, npepts=7, FCcut=10):
    bid = pd.read_csv("msresist/data/BioID/BioID.csv")
    genes = [s.split("_")[0] for i, s in enumerate(bid["Gene"])]
    bid = bid.drop(["Protein", "Gene"], axis=1)
    bid.insert(0, "Gene", genes)
    XIDX = np.any(bid.iloc[:, 1:].values > npepts, axis=1)
    bid = bid.iloc[XIDX, :]
    bid = bid.replace(0, 0.00001)
    bid.iloc[:, 1:] = np.log(bid.iloc[:, 1:])
    bid = bid.set_index("Gene").T.reset_index()
    bid["index"] = [s.split(".")[0] for s in bid["index"]]
    nc = bid.groupby("index").mean().loc["Bio"]
    fc = bid.copy()
    fc.iloc[:, 1:] += nc.abs()
    XIDX = np.any(fc.iloc[:, 1:].values >= FCcut, axis=0)
    XIDX = np.insert(XIDX, 0, True)
    bid = bid.iloc[:, XIDX]
    if linear:
        bid = Linear(bid, bid.columns[1:])
    return bid


def bootPCA(d, n_components, lIDX, method="PCA", n_boots=100):
    """ Compute PCA scores and loadings including bootstrap variance of the estimates. """
    bootScor, bootLoad = [], []
    data_headers = list(d.select_dtypes(include=['float64']).columns)
    sIDX = list(d.select_dtypes(include=['object']).columns)
    for _ in range(n_boots):
        xIDX = range(d.shape[0])
        resamp = resample(xIDX, replace=True)
        bootdf = d.iloc[resamp, :].groupby(sIDX).mean().reset_index()
        data = bootdf[data_headers]

        if method == "PCA":
            bootdf[data_headers] = StandardScaler().fit_transform(data)
            red = PCA(n_components=n_components)
            dScor = red.fit_transform(data.values)
            varExp = np.round(red.explained_variance_ratio_, 2)

        elif method == "NMF":
            varExp = []
            for i in range(1, n_components + 1):
                red = NMF(n_components=i, max_iter=10000, solver="mu", beta_loss="frobenius", init='nndsvdar').fit(data.values)
                dScor = red.transform(data)
                varExp.append(r2_score(data, red.inverse_transform(dScor)))

        dScor, dLoad = pca_dfs(dScor, red.components_, bootdf, n_components, sIDX, lIDX)
        bootScor.append(dScor)
        bootLoad.append(dLoad)

    bootScor = pd.concat(bootScor)
    bootScor_m = bootScor.groupby(sIDX).mean().reset_index()
    bootScor_sd = bootScor.groupby(sIDX).sem().reset_index()

    bootLoad = pd.concat(bootLoad)
    bootLoad_m = bootLoad.groupby(lIDX).mean().reset_index()
    bootLoad_sd = bootLoad.groupby(lIDX).sem().reset_index()

    return bootScor_m, bootScor_sd, bootLoad_m, bootLoad_sd, bootScor, varExp


def plotBootPCA(ax, means, stds, varExp, title=False, X="PC1", Y="PC2", LegOut=False, annotate=False, colors=False):
    """ Plot Scores and Loadings. """
    sIDX = list(means.select_dtypes(include=['object']).columns)
    hue = sIDX[0]
    style = None
    if len(sIDX) == 2:
        style = sIDX[1]

    ax.errorbar(means[X], means[Y], xerr=stds[X], yerr=stds[Y],
                linestyle="", elinewidth=0.2, capsize=2, capthick=0.2, ecolor='k')

    if colors:
        pal = sns.xkcd_palette(colors)
        p1 = sns.scatterplot(x=X, y=Y, data=means, hue=hue, style=style, ax=ax,
                             palette=pal, markers=["o", "X", "d", "*"], **{'linewidth': .5, 'edgecolor': "k"}, s=55)


    if not colors:
        p1 = sns.scatterplot(x=X, y=Y, data=means, hue=hue, style=style, ax=ax,
                             markers=["o", "X", "d", "*"], **{'linewidth': .5, 'edgecolor': "k"}, s=55)

    if LegOut:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    ax.set_title(title)

    if annotate:
        for idx, txt in enumerate(means[sIDX[0]]):
            p1.text(means[X][idx], means[Y][idx], txt,
                    horizontalalignment='left', color='black', size="xx-small", fontweight="light")

    ax.set_xlabel(str(X) + "(" + str(int(varExp[int(X[-1]) - 1] * 100)) + "%)", fontsize=10)
    ax.set_ylabel(str(Y) + "(" + str(int(varExp[int(Y[-1]) - 1] * 100)) + "%)", fontsize=10)
