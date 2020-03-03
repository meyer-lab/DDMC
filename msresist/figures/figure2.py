"""
This creates Figure 2.
"""
import os
import pandas as pd
import numpy as np
import scipy as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .common import subplotLabel, getSetup
from sklearn.model_selection import cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from msresist.clustering import MassSpecClustering
from msresist.parameter_tuning import MSclusPLSR_tuning, kmeansPLSR_tuning
from msresist.plsr import Q2Y_across_components, R2Y_across_components
from msresist.sequence_analysis import preprocess_seqs
import matplotlib.colors as colors
import matplotlib.cm as cm
from msresist.pre_processing import preprocessing, MergeDfbyMean


path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 3))

    # blank out first axis for cartoon
    ax[0].axis('off')

    # Cell Viability
    cv1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR1.csv").iloc[:, 1:]
    cv1_ab = cv1.loc[:, cv1.columns.str.contains('-A/E')]
    cv2 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR2.csv').iloc[:, 1:]
    cv2_ab = cv2.loc[:, cv2.columns.str.contains('-A/E')]
    cv3 = pd.read_csv('msresist/data/Phenotypic_data/AXLmutants/20200130-AXLmutantsPhase_MeanTRs_BR3.csv').iloc[:, 1:]
    cv3_ab = cv3.loc[:, cv2.columns.str.contains('-A/E')]

    for ii in range(0, cv2_ab.columns.size):
        cv1_ab.iloc[:, ii] /= cv1_ab.iloc[0, ii]
        cv2_ab.iloc[:, ii] /= cv2_ab.iloc[0, ii]
        cv3_ab.iloc[:, ii] /= cv3_ab.iloc[0, ii]

    cv = pd.concat([cv1_ab, cv2_ab, cv3_ab], axis=0)
    cv.insert(0, "Elapsed", cv1.iloc[:, 0])
    cv = MergeDfbyMean(cv, cv1_ab.columns, "Elapsed").reset_index()
    cv = cv[cv["Elapsed"] == 120].iloc[0, 1:]
    v = cv[["PC9-A/E", "AXL KO-A/E", "Kdead-A/E", "Kin-A/E", "M4-A/E", "M5-A/E", "M7-A/E", "M10-A/E", "M11-A/E", "M15-A/E"]]

    # Phosphorylation data
    X = preprocessing(Axlmuts_ErlF154=True, motifs=True, Vfilter=False, FCfilter=False, log2T=True, mc_row=True)
    X = preprocess_seqs(X, "Y").sort_values(by="Protein")

    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    lines = ["PC9", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]
    d.index = lines

    distance_method = "Binomial"
    ncl = 3
    GMMweight = 0.75
    b = ncl + 1

    ncomp = 2
    mixedCl_plsr = Pipeline([('mixedCl', MassSpecClustering(i, ncl, GMMweight=GMMweight, distance_method=distance_method)), ('plsr', PLSRegression(ncomp))])
    fit = mixedCl_plsr.fit(d, v)

    plotR2YQ2Y(ax[1], mixedCl_plsr, d, v, cv=2, b=b)

    centers = mixedCl_plsr.named_steps.mixedCl.transform(d)

    plotMeasuredVsPredicted(ax[2], mixedCl_plsr, d, v)

    plotScoresLoadings(ax[3:5], fit, centers, v, ncl, lines, CV=2)

    plotclusteraverages(ax[5], centers.T, lines)

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotR2YQ2Y(ax, model, X, Y, cv, b=3):
    Q2Y = Q2Y_across_components(model, X, Y, cv, b)
    R2Y = R2Y_across_components(model, X, Y, cv, b)

    range_ = np.arange(1, b)

    ax.bar(range_ + 0.15, Q2Y, width=0.3, align='center', label='Q2Y', color="darkblue")
    ax.bar(range_ - 0.15, R2Y, width=0.3, align='center', label='R2Y', color="black")
    ax.set_title("R2Y/Q2Y Cell Viability")
    ax.set_xticks(range_)
    ax.set_xlabel("Number of Components")
    ax.legend(loc=0)


def plotMixedClusteringPLSR_GridSearch(ax, X, info, Y, distance_method):
    CVresults_max, CVresults_min, best_params = MSclusPLSR_tuning(X, info, Y, distance_method)
    ncl_GMMweight_ncomp = CVresults_min.sort_values(by="Ranking").iloc[:21, :]

    labels = []
    for ii in range(ncl_GMMweight_ncomp.shape[0]):
        labels.append(str(ncl_GMMweight_ncomp.iloc[ii, 1]) + "|" + str(ncl_GMMweight_ncomp.iloc[ii, 2]))

    width = 0.20
    ax.bar(np.arange(ncl_GMMweight_ncomp.shape[0]), np.abs(ncl_GMMweight_ncomp.iloc[:, 3]), width, edgecolor='black', color='g')
    ax.set_xticks(np.arange(ncl_GMMweight_ncomp.shape[0]))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Number of Clusters | GMM Weight")
    ax.set_ylabel("Mean-Squared Error (MSE)")
    ax.set_title("Top20 Hyperparameter Combinations (N Components=2)")


def plotMeasuredVsPredicted(ax, plsr_model, X, Y):
    """ Plot exprimentally-measured vs PLSR-predicted values. """
    Y_predictions = list(np.squeeze(cross_val_predict(plsr_model, X, Y, cv=Y.size, n_jobs=-1)))
    Y = list(Y)
    ax.scatter(Y, Y_predictions)
    ax.plot(np.unique(Y), np.poly1d(np.polyfit(Y, Y_predictions, 1))(np.unique(Y)), color="r")
    ax.set(title="Correlation Measured vs Predicted", xlabel="Actual Y", ylabel="Predicted Y")
    ax.set_title("Correlation Measured vs Predicted")
    ax.set_xlabel("Measured Cell Viability")
    ax.set_ylabel("Predicted Cell Viability")
    ax.set_xlim([1, np.max(Y) * 1.2])
    ax.set_ylim([1, np.max(Y) * 1.2])
    coeff, _ = sp.stats.pearsonr(list(Y_predictions), list(Y))
    textstr = "$r$ = " + str(np.round(coeff, 4))
    props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
    ax.text(0.80, 0.09, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, CV):
    """ Plot scores and loadings plot """
    if CV == 1:
        X_scores, _ = model.transform(X, Y)
        PC1_xload, PC2_xload = model.x_loadings_[:, 0], model.x_loadings_[:, 1]
        PC1_yload, PC2_yload = model.y_loadings_[:, 0], model.y_loadings_[:, 1]

    if CV == 2:
        X_scores, _ = model.named_steps.plsr.transform(X, Y)
        PC1_xload, PC2_xload = model.named_steps.plsr.x_loadings_[:, 0], model.named_steps.plsr.x_loadings_[:, 1]
        PC1_yload, PC2_yload = model.named_steps.plsr.y_loadings_[:, 0], model.named_steps.plsr.y_loadings_[:, 1]

    PC1_scores, PC2_scores = X_scores[:, 0], X_scores[:, 1]
    colors_ = cm.rainbow(np.linspace(0, 1, ncl))

    # Scores
    ax[0].scatter(PC1_scores, PC2_scores)
    for j, txt in enumerate(treatments):
        ax[0].annotate(txt, (PC1_scores[j], PC2_scores[j]))
    ax[0].set_title('PLSR Model Scores')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    ax[0].axhline(y=0, color='0.25', linestyle='--')
    ax[0].axvline(x=0, color='0.25', linestyle='--')

    spacer = 0.5
    ax[0].set_xlim([(-1 * max(np.abs(PC1_scores))) - spacer, max(np.abs(PC1_scores)) + spacer])
    ax[0].set_ylim([(-1 * max(np.abs(PC2_scores))) - spacer, max(np.abs(PC2_scores)) + spacer])

    # Loadings
    numbered = []
    list(map(lambda v: numbered.append(str(v + 1)), range(ncl)))
    for i, txt in enumerate(numbered):
        ax[1].annotate(txt, (PC1_xload[i], PC2_xload[i]))
    ax[1].annotate("Cell Viability", (PC1_yload + 0.05, PC2_yload - 0.05))
    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].scatter(PC1_yload, PC2_yload, color='#000000', marker='D', label='Cell Viability')
    ax[1].set_title('PLSR Model Loadings (Averaged Clusters)')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].axhline(y=0, color='0.25', linestyle='--')
    ax[1].axvline(x=0, color='0.25', linestyle='--')

    spacer = 0.5
    ax[1].set_xlim([(-1 * max(np.abs(list(PC1_xload) + list(PC1_yload)))) - spacer, max(np.abs(list(PC1_xload) + list(PC1_yload))) + spacer])
    ax[1].set_ylim([(-1 * max(np.abs(list(PC2_xload) + list(PC2_yload)))) - spacer, max(np.abs(list(PC2_xload) + list(PC2_yload))) + spacer])


def plotScoresLoadings_plotly(X, labels, Y, ncomp, loc=False):
    """ Interactive PLSR plot. Note that this works best by pre-defining the dataframe's
    indices which will serve as labels for each dot in the plot. """

    plsr = PLSRegression(ncomp)
    X_scores, _ = plsr.fit_transform(X, Y)
    scores = pd.concat([pd.DataFrame(X_scores[:, 0]),
                        pd.DataFrame(X_scores[:, 1])], axis=1)
    scores.index = X.index
    scores.columns = ["PC1", "PC2"]

    xloads = pd.concat([pd.DataFrame(plsr.x_loadings_[:, 0]),
                        pd.DataFrame(plsr.x_loadings_[:, 1])], axis=1)
    yloads = pd.concat([pd.DataFrame(plsr.y_loadings_[:, 0]),
                        pd.DataFrame(plsr.y_loadings_[:, 1])], axis=1)

    xloads.index, yloads.index = X.columns, yloads.index.rename("Cell Viability")
    xloads.columns, yloads.columns = [["PC1", "PC2"]] * 2

    if loc:
        print(loadings.loc[loc])

    colors_ = ["black", "red", "blue", "lightgoldenrodyellow", "brown", "cyan", "orange", "gray"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PLSR Scores", "PLSR Loadings"))
    fig.add_trace(
        go.Scatter(
            mode='markers+text',
            x=scores["PC1"],
            y=scores["PC2"],
            text=scores.index,
            textposition="top center",
            textfont=dict(
                size=10,
                color="black"),
            marker=dict(
                color='blue',
                size=8,
                line=dict(
                    color='black',
                    width=1))),
        row=1, col=1)

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=xloads["PC1"],
            y=xloads["PC2"],
            opacity=0.7,
            text=["Protein: " + xloads.index[i][0] + "  Pos: " + xloads.index[i][1] for i in range(len(xloads.index))],
            marker=dict(
                color=[colors_[i] for i in labels],
                size=8,
                line=dict(
                    color='black',
                    width=1))),
        row=1, col=2)

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=yloads["PC1"],
            y=yloads["PC2"],
            opacity=0.7,
            text=yloads.index.name,
            marker=dict(
                color='green',
                size=10,
                line=dict(
                    color='black',
                    width=1))),
        row=1, col=2)

    fig.update_layout(
        height=500,
        width=1000,
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(showgrid=False))
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=1)
    fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)

    fig.show()


def plotclusteraverages(ax, centers, treatments):
    """Plot the cluster averages across conditions """
    colors_ = cm.rainbow(np.linspace(0, 1, centers.shape[0]))

    for i in range(centers.shape[0]):
        ax.plot(centers.iloc[i, :], marker='o', label="cluster " + str(i + 1), color=colors_[i])

    ax.set_xticks(np.arange(centers.shape[1]))
    ax.set_xticklabels(treatments, rotation=45)
    ax.set_ylabel("Normalized Signal", fontsize=12)
    ax.legend()


def plotKmeansPLSR_GridSearch(ax, X, Y):
    CVresults_max, CVresults_min, best_params = kmeansPLSR_tuning(X, Y)
    twoC = np.abs(CVresults_min.iloc[:2, 3])
    threeC = np.abs(CVresults_min.iloc[2:5, 3])
    fourC = np.abs(CVresults_min.iloc[5:9, 3])
    fiveC = np.abs(CVresults_min.iloc[9:14, 3])
    sixC = np.abs(CVresults_min.iloc[14:20, 3])

    width = 1
    groupgap = 1

    x1 = np.arange(len(twoC))
    x2 = np.arange(len(threeC)) + groupgap + len(twoC)
    x3 = np.arange(len(fourC)) + groupgap * 2 + len(twoC) + len(threeC)
    x4 = np.arange(len(fiveC)) + groupgap * 3 + len(twoC) + len(threeC) + len(fourC)
    x5 = np.arange(len(sixC)) + groupgap * 4 + len(twoC) + len(threeC) + len(fourC) + len(fiveC)

    ax.bar(x1, twoC, width, edgecolor='black', color="g")
    ax.bar(x2, threeC, width, edgecolor='black', color="g")
    ax.bar(x3, fourC, width, edgecolor='black', color="g")
    ax.bar(x4, fiveC, width, edgecolor="black", color="g")
    ax.bar(x5, sixC, width, edgecolor="black", color="g")

    comps = []
    for ii in range(2, 7):
        comps.append(list(np.arange(1, ii + 1)))
    flattened = [nr for cluster in comps for nr in cluster]

    ax.set_xticks(np.concatenate((x1, x2, x3, x4, x5)))
    ax.set_xticklabels(flattened, fontsize=10)
    ax.set_xlabel("Number of Components per Cluster")
    ax.set_ylabel("Mean-Squared Error (MSE)")


def plotCoClusteringGridSearch(ax, grid_results):
    labels = []
    for ii in range(grid_results.shape[0]):
        labels.append(str(grid_results.iloc[ii, 1]) + 
                      "|" + str(grid_results.iloc[ii, 2]) + 
                      "|" + str(grid_results.iloc[ii, 3]))


    fig, ax = plt.subplots(1,1,figsize=(25,10))

    width = 0.5
    ax.bar(np.arange(grid_results.shape[0]), np.abs(grid_results.iloc[:, 4]), width, edgecolor='black', color='g')
    ax.set_xticks(np.arange(grid_results.shape[0]))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("#Clusters | #Components | GMM Weight", fontsize=16)
    ax.set_ylabel("Mean-Squared Error (MSE)", fontsize=16)
    ax.set_title("Top20 Hyperparameter Combinations", fontsize=20)