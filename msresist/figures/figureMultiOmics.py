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

    p_dfrot = []
    for idx, gene in enumerate(predictor_prot.columns):
        p_dfrot.append([])
        for clust in response_prot.columns:
            cov = np.cov(list(predictor_prot[gene].dropna()),list(response_prot[clust].loc[predictor_prot[gene].dropna().index]))
            p_dfrot[idx].append(cov[0][1]/((cov[0][0]*cov[1][1])**.5))
    p_dfrot = np.asarray(p_dfrot).T

    corr_out= pd.DataFrame(corr, columns = predictor.columns, index = response.columns)
    corr_out = corr_out.append(mRNA_data['geneSymbol'])
    corr_out.to_csv('mRNA_Cluster_Correlations.csv')
    p_dfrot_out= pd.DataFrame(p_dfrot, columns = predictor_prot.columns, index = response_prot.columns)
    p_dfrot_out = p_dfrot_out.append(prot_data['geneSymbol'])
    p_dfrot_out.to_csv('prot_Cluster_Correlations.csv')

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 1))

    # Add subplot labels
    #subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})
    


    m_df= pd.read_csv('mRNA_Cluster_Correlations.csv')
    p_df = pd.read_csv('prot_Cluster_Correlations.csv')
    m_df = m_df[m_df.index!=24]
    p_df = p_df[p_df.index!=24]
    corr_1 = np.array([], dtype = float)
    corr_2 = np.array([], dtype = float)

    for col_m in m_df.columns[1:]:
        corr_1 = np.concatenate((corr_1,np.asarray(m_df[col_m][m_df.index<12], dtype = float)))
        corr_2 = np.concatenate((corr_2,np.asarray(m_df[col_m][m_df.index>=12], dtype = float)))
    

    for col_p in p_df.columns[1:]:
        corr_1 = np.concatenate((corr_1,np.asarray(p_df[col_p][p_df.index<12], dtype = float)))
        corr_2 = np.concatenate((corr_2,np.asarray(p_df[col_p][p_df.index>=12], dtype = float)))
    
    mol = ["mRNA" if idx < (len(m_df.columns)-1)*12 else "protein"  for idx  in range((len(m_df.columns)+len(p_df.columns)-2)*12)]   
    clust_1 = np.arange(0,len(corr_1))%12+1           
    clust_2 = np.arange(0,len(corr_2))%12+13  
    group1 = pd.DataFrame(np.asarray([corr_1, clust_1, mol]).T, columns = ['Correlation','Cluster', 'Molecule'])
    group2 = pd.DataFrame(np.asarray([corr_2, clust_2, mol]).T, columns = ['Correlation','Cluster', 'Molecule'])
    group1 = group1.astype({'Correlation':'float64', 'Cluster': 'str', 'Molecule': 'str'})
    group2 = group2.astype({'Correlation':'float64', 'Cluster': 'str', 'Molecule': 'str'})

    sns.violinplot(x = "Cluster", y = "Correlation",data = group1, ax = ax[0], hue = "Molecule", split=True)
    sns.violinplot(x = "Cluster", y = "Correlation",data = group2, ax = ax[1], hue = "Molecule", split=True)

    return f