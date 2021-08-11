import pandas as pd
import copy
import numpy as np
import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from .common import subplotLabel, getSetup

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
    ax, f = getSetup((12, 10), (4, 1))

    # Add subplot labels
    #subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
    


    corr_out = pd.read_csv('mRNA_Cluster_Correlations.csv')
    corr_out = corr_out[corr_out.index!=24]
    corr_temp = np.array([], dtype = float)
    clust_temp = np.array([],int)
    corr_temp_2 = np.array([], dtype = float)
    clust_temp_2 = np.array([],int)
    for col in corr_out.columns:
        if col != corr_out.columns[0]:
            corr_temp = np.concatenate((corr_temp,np.asarray(corr_out[col][corr_out.index<12], dtype = float)))
            clust_temp = np.concatenate((clust_temp,np.arange(1,13)))
            corr_temp_2 = np.concatenate((corr_temp_2,np.asarray(corr_out[col][corr_out.index>=12], dtype = float)))
            clust_temp_2 = np.concatenate((clust_temp_2,np.arange(13,25)))
    to_plot = pd.DataFrame(np.asarray([corr_temp, clust_temp]).T, columns = ['corr','clust'])
    to_plot_2 = pd.DataFrame(np.asarray([corr_temp_2, clust_temp_2]).T, columns = ['corr','clust'])

    corr_prot_out = pd.read_csv('prot_Cluster_Correlations.csv')
    corr_prot_out = corr_prot_out[corr_prot_out.index!=24]
    corr_prot_temp = np.array([], dtype = float)
    clust_prot_temp = np.array([],int)
    corr_prot_temp_2 = np.array([], dtype = float)
    clust_prot_temp_2 = np.array([],int)
    for col in corr_prot_out.columns:
        if col != corr_prot_out.columns[0]:
            corr_prot_temp = np.concatenate((corr_prot_temp,np.asarray(corr_prot_out[col][corr_prot_out.index<12], dtype = float)))
            clust_prot_temp = np.concatenate((clust_prot_temp,np.arange(1,13)))
            corr_prot_temp_2 = np.concatenate((corr_prot_temp_2,np.asarray(corr_prot_out[col][corr_prot_out.index>=12], dtype = float)))
            clust_prot_temp_2 = np.concatenate((clust_prot_temp_2,np.arange(13,25)))
    to_plot_prot = pd.DataFrame(np.asarray([corr_prot_temp, clust_prot_temp]).T, columns = ['corr','clust'])
    to_plot_prot_2 = pd.DataFrame(np.asarray([corr_prot_temp_2, clust_prot_temp_2]).T, columns = ['corr','clust'])
   
    ax[0].set_title('mRNA')
    ax[2].set_title('Protein')
    sns.violinplot(x = "clust", y = "corr",data = to_plot, ax = ax[0])
    sns.violinplot(x = "clust", y = "corr",data = to_plot_2, ax = ax[1])
    sns.violinplot(x = "clust", y = "corr",data = to_plot_prot, ax = ax[2])
    sns.violinplot(x = "clust", y = "corr",data = to_plot_prot_2, ax = ax[3])
    return f