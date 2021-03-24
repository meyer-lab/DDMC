"""
This creates Supplemental Figure 3: Prediction performance across different plsr models using different clustering strategies
"""
import pickle
import seaborn as sns
from .common import subplotLabel, getSetup
from pomegranate import GeneralMixtureModel, NormalDistribution
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from ..clustering import MassSpecClustering
from ..pre_processing import preprocessing
from .figure1 import import_phenotype_data, formatPhenotypesForModeling
from .figure3 import plotR2YQ2Y, plotActualVsPredicted

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 11), (3, 4), multz={0:1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # mass spec data
    X = preprocessing(Axlmuts_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    d = X.select_dtypes(include=['float64']).T
    i = X.select_dtypes(include=['object'])

    # Import phenotypes
    cv = import_phenotype_data(phenotype="Cell Viability")
    red = import_phenotype_data(phenotype="Cell Death")
    sw = import_phenotype_data(phenotype="Migration")
    c = import_phenotype_data(phenotype="Island")
    y = formatPhenotypesForModeling(cv, red, sw, c)
    y = y[y["Treatment"] == "A/E"].drop("Treatment", axis=1).set_index("Lines")

    # Pipeline diagram
    ax[0].axis("off")

    # No clustering
    plotR2YQ2Y(ax[1], PLSRegression(n_components=2, scale=True), d, y, 6, color="grey", title="Raw data")
    plotActualVsPredicted(ax[2], PLSRegression(n_components=3), d, y, y_pred="cross-validation", color="grey", type="bar", title="Raw data")

    ncl = 5
    # k-means
    labels = KMeans(n_clusters=ncl).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T
    plotR2YQ2Y(ax[3], PLSRegression(n_components=2), c_kmeans, y, 6, color="darkred", title="k-means")
    plotActualVsPredicted(ax[4], PLSRegression(n_components=2), c_kmeans, y, y_pred="cross-validation", color="darkred", type="bar", title="k-means")

    # GMM
    gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d.T, n_components=ncl, n_jobs=-1)
    x_ = X.copy()
    x_["Cluster"] = gmm.predict(d.T)
    c_gmm = x_.groupby("Cluster").mean().T
    plotR2YQ2Y(ax[5], PLSRegression(n_components=3), c_gmm, y, 6, color="olive", title="GMM")
    plotActualVsPredicted(ax[6], PLSRegression(n_components=3), c_gmm, y, y_pred="cross-validation", color="olive", type="bar", title="GMM")

    # DDMC (w=50)
    model = MassSpecClustering(i, ncl=5, SeqWeight=20, distance_method="PAM250").fit(d, y)
    centers = model.transform()
    plotR2YQ2Y(ax[7], PLSRegression(n_components=3), centers, y, model.ncl + 1, title="DDMC Sequence", color="orange")
    plotActualVsPredicted(ax[8], PLSRegression(n_components=2), centers, y, y_pred="cross-validation", color="orange", type="bar", title="DDMC Sequence")

    # DDMC (w=2)
    with open('msresist/data/pickled_models/AXLmodel_PAM250_W2_5CL', 'rb') as m:
        model = pickle.load(m)[0]
    centers = model.transform()
    plotR2YQ2Y(ax[9], PLSRegression(n_components=4), centers, y, model.ncl + 1, title="DDMC mix")
    plotActualVsPredicted(ax[10], PLSRegression(n_components=4), centers, y, y_pred="cross-validation", color="darkblue", type="bar", title="DDMC mix")

    return f
