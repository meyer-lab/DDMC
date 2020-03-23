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
import seaborn as sns
from msresist.pre_processing import preprocessing, MergeDfbyMean, cv_pre, cm_pre
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

path = os.path.dirname(os.path.abspath(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 10), (3, 4))

    # blank out first axis for cartoon
#     ax[0].axis('off')

    # Import Cell Viability data
    all_lines = ["PC9-A/E", "AXL KO-A/E", "Kdead-A/E", "Kin-A/E", "M4-A/E", "M5-A/E", "M7-A/E", "M10-A/E", "M11-A/E", "M15-A/E"]
    mut_lines = all_lines[1:]
    g_lines = all_lines[2:]

    cv1 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR1_Phase.csv")
    cv2 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR2_Phase.csv")
    cv3 = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/CellViability/Phase/BR3_Phase.csv")
    itp = 24
    ftp = 120
    tr = "A/E"

    v = cv_pre(cv1, cv2, cv3, tr, itp, ftp, all_lines)

    #Import Apoptosis data
    
    
    #Import Cell Migration data
    all_lines = ["PC9 A/E", "KO A/E", "KD A/E", "KIN A/E", "M4 A/E", "M5 A/E", "M7 A/E", "M10 A/E", "M11 A/E", "M15 A/E"]
    mut_lines = all_lines[1:]
    g_lines = all_lines[2:]
    
    rwd = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/03062020-AXLmuts_EMT_RWD_Collagen_BR1.csv")
    rwdg = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/03062020-AXLmuts_EMT_GreenRWD_Collagen_BR1.csv")
    wc = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/03062020-AXLmuts_EMT_WC_Collagen_BR1.csv")

    ftp = 24

    m = cm_pre(wc, tr, ftp, all_lines)
    m.index = v.index

    #Build Y matrix
    y = pd.concat([v, m], axis=1)
    y.columns = ["Viability", "Migration"]

    # Import Phosphorylation data and build X matrix
    X = preprocessing(Axlmuts_ErlF154=True, motifs=True, Vfilter=False, FCfilter=False, log2T=True, mc_row=True)
    X = preprocess_seqs(X, "Y").sort_values(by="Protein")

    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    all_lines = ["PC9", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"] 
    mut_lines = all_lines[1:]
    g_lines = all_lines[2:]
    
    d.index = all_lines
    
    #### ------ Co-clustering and PLSR model ------ ####
    # Cross-validation 1: leaving one condition out across fixed clusters - fitting PLSR each time
    distance_method = "PAM250"
    ncl = 11
    GMMweight = 5
    ncomp = 2

    MSC = MassSpecClustering(i, ncl, GMMweight=GMMweight, distance_method=distance_method).fit(d, y)
    centers = MSC.transform(d)
    plsr = PLSRegression(n_components=2)

    plotR2YQ2Y(ax[0], plsr, centers, y, cv=1, b=ncl+1)
    plotMeasuredVsPredicted(ax[1:3], plsr, centers, y)
    plotScoresLoadings(ax[3:5], plsr.fit(centers, y), centers, y, ncl, all_lines, 1)
#     plotclusteraverages(ax[5], MSC.transform(d).T, all_lines)

    # Cross-validation 2: leaving one condition out across peptides - fitting entire model pipeline each time
    gs = pd.read_csv("msresist/data/Model/20200320-GridSearch_pam250_CVWC_wPC9.csv")
    CoCl_plsr = Pipeline([('CoCl', MassSpecClustering(i, ncl, GMMweight=GMMweight, distance_method=distance_method)), ('plsr', PLSRegression(ncomp))])
    fit = CoCl_plsr.fit(d, y)
    centers = CoCl_plsr.named_steps.CoCl.transform(d)

    plotR2YQ2Y(ax[5], CoCl_plsr, d, y, cv=2, b=ncl+1)
    plotGridSearch(ax[6], gs)
    plotMeasuredVsPredicted(ax[7:9], CoCl_plsr, d, y)
    plotScoresLoadings(ax[9:11], fit, centers, y, ncl, all_lines, 2)
    plotclusteraverages(ax[11], centers.T, all_lines)

    # Add subplot labels
    subplotLabel(ax)

    return f


def plotGridSearch(ax, gs):
    """ Plot gridsearch results by ranking. """
    ax = sns.barplot(x="Ranking", y="mean_test_scores", data=np.abs(gs.iloc[:20, :]), ax=ax, **{"linewidth":.5}, **{"edgecolor":"black"})
    ax.set_title("Hyperaparameter Search")
    ax.set_xticklabels(np.arange(1, 21))
    ax.set_ylabel("Mean Squared Error")


def plotR2YQ2Y(ax, model, X, Y, cv, b=3):
    """ Plot R2Y/Q2Y variance explained by each component. """
    Q2Y = Q2Y_across_components(model, X, Y, cv, b)
    R2Y = R2Y_across_components(model, X, Y, cv, b)
    
    range_ = np.arange(1, b)

    ax.bar(range_ + 0.15, Q2Y, width=0.3, align='center', label='Q2Y', color="darkblue")
    ax.bar(range_ - 0.15, R2Y, width=0.3, align='center', label='R2Y', color="black")
    ax.set_title("R2Y/Q2Y - Cross-validation strategy: " + str(cv))
    ax.set_xticks(range_)
    ax.set_xlabel("Number of Components")
    ax.set_xlabel("Variance")
    ax.legend(loc=0)


def plotMeasuredVsPredicted(ax, plsr_model, X, Y):
    """ Plot exprimentally-measured vs PLSR-predicted values. """
    Y_predictions = cross_val_predict(plsr_model, X, Y, cv=Y.shape[0])
    for i, label in enumerate(Y.columns):
        y = Y.iloc[:, i]
        ypred = Y_predictions[:, i]
        ax[i].scatter(y, ypred)
        ax[i].plot(np.unique(y), np.poly1d(np.polyfit(y, ypred, 1))(np.unique(y)), color="r")
        ax[i].set(title=label, xlabel="Measured", ylabel="Predicted")
        ax[i].set_xlim([np.min(y) * 0.9, np.max(y) * 1.1])
        ax[i].set_ylim([np.min(y) * 0.9, np.max(y) * 1.1])

        #Add correlation coefficient
        coeff, _ = sp.stats.pearsonr(ypred, y)
        textstr = "$r$ = " + str(np.round(coeff, 4))
        props = dict(boxstyle='square', facecolor='none', alpha=0.5, edgecolor='black')
        ax[i].text(0.80, 0.09, textstr, transform=ax[i].transAxes, fontsize=10, verticalalignment='top', bbox=props)


def plotScoresLoadings(ax, model, X, Y, ncl, treatments, cv):
    if cv ==1:
        X_scores, _ = model.transform(X, Y)
        PC1_xload, PC2_xload = model.x_loadings_[:, 0], model.x_loadings_[:, 1]
        PC1_yload, PC2_yload = model.y_loadings_[:, 0], model.y_loadings_[:, 1]
    
    if cv == 2:
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
    ax[1].annotate("Viability", (PC1_yload[0] + 0.05, PC2_yload[0] - 0.05))
    ax[1].annotate("Migration", (PC1_yload[1] + 0.05, PC2_yload[1] - 0.05))
    ax[1].scatter(PC1_xload, PC2_xload, c=np.arange(ncl), cmap=colors.ListedColormap(colors_))
    ax[1].scatter(PC1_yload[0], PC2_yload[0], color='grey', marker='x')
    ax[1].scatter(PC1_yload[1], PC2_yload[1], color='#000000', marker='D')
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
