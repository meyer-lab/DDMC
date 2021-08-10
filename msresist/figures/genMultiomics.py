import pandas as pd
import copy
import numpy as np
import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def gen_csvs():
    clust_data = pd.read_csv('CPTAC_LUAD_CL24_W15_TMT2_Centers.csv')
    prot_data = pd.read_excel('mmc3.xlsx', 1)
    mRNA_data = pd.read_excel('mmc2.xlsx', sheet_name='Table S2E')

    #prot import
    prot_data.columns = np.asarray(prot_data.iloc[1])
    prot_data = prot_data.drop([0,1])
    prot_data = prot_data[prot_data['geneSymbol'] != 'na']
    prot_data.index =np.array(prot_data['id']).T[0]

    #mRNA import
    mRNA_data.columns = np.asarray(mRNA_data.iloc[1])
    mRNA_data = mRNA_data.drop([0,1])
    mRNA_data = mRNA_data[mRNA_data['geneSymbol'] != 'na']
    mRNA_data.index =np.array(mRNA_data['gene_id'])

    #clust import
    clust_data.index = clust_data['Patient_ID']
    clust_data.drop(clust_data.columns[0:2],axis = 1, inplace = True)
    clust_data = clust_data.transpose()

    #construct predictor+response
    predictor = mRNA_data[mRNA_data.columns[6:]].T
    response = clust_data[mRNA_data.columns[6:]].T
    predictor_prot = prot_data[prot_data.columns[17:]].T
    response_prot = clust_data[prot_data.columns[17:]].T

    #get corrs
    corr = []
    for idx, gene in enumerate(predictor.columns):
        corr.append([])
        for clust in response.columns:
            cov = np.cov(list(predictor[gene].dropna()),list(response[clust].loc[predictor[gene].dropna().index]))
            corr[idx].append(cov[0][1]/((cov[0][0]*cov[1][1])**.5))
    corr = np.asarray(corr).T

    corr_prot = []
    for idx, gene in enumerate(predictor_prot.columns):
        corr_prot.append([])
        for clust in response_prot.columns:
            cov = np.cov(list(predictor_prot[gene].dropna()),list(response_prot[clust].loc[predictor_prot[gene].dropna().index]))
            corr_prot[idx].append(cov[0][1]/((cov[0][0]*cov[1][1])**.5))
    corr_prot = np.asarray(corr_prot).T

    corr_out= pd.DataFrame(corr, columns = predictor.columns, index = response.columns)
    corr_out = corr_out.append(mRNA_data['geneSymbol'])
    corr_out.to_csv('mRNA_Cluster_Correlations.csv')
    corr_prot_out= pd.DataFrame(corr_prot, columns = predictor_prot.columns, index = response_prot.columns)
    corr_prot_out = corr_prot_out.append(prot_data['geneSymbol'])
    corr_prot_out.to_csv('prot_Cluster_Correlations.csv')

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 5), (2, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load DDMC
    with open("msresist/data/pickled_models/AXLmodel_PAM250_W2-5_5CL", "rb") as m:
        model = pickle.load(m)
    centers = model.transform()
    lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    # Centers
    plotCenters(ax[:5], model, lines)

    # Plot motifs
    pssms = model.pssms(PsP_background=True)
    plotMotifs([pssms[0], pssms[1], pssms[2], pssms[3], pssms[4]], axes=ax[5:10], titles=["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"], yaxis=[0, 11])

    return f