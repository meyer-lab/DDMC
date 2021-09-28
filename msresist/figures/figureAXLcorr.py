"""
Creates plots related to correlating phosphoclusters of LUAD patients and AXL expression
"""

import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

rnaC = pd.read_csv("msresist/data/MS/CPTAC/Omics results/mRNA_Cluster_Correlations.csv").drop("Unnamed: 0", axis=1)
protC = pd.read_csv("msresist/data/MS/CPTAC/Omics results/prot_Cluster_Correlations.csv").drop("Unnamed: 0", axis=1)

#Change column labels to Symbol genes
rnaC.columns = rnaC.iloc[-1, :]
rnaC = rnaC.iloc[:-1, :]
protC.columns = protC.iloc[-1, :]
protC = protC.iloc[:-1, :]

def heatmap_ClusterVsTargets_Corr(targets, omic="Protein", title=False):
    """Plot correlations between clusters and targets"""
    tar = pd.DataFrame()
    for t in targets:
        try:
            if omic == "Protein":
                tar[t] = protC[t]
            elif omic == "RNA":
                tar[t] = rnaC[t]
        except:
            print(t + " not present in the data set")
            continue
    tar = tar.astype(float)
    tar.index += 1
    g = sns.clustermap(tar.astype(float), figsize=(5, 10), cmap="bwr")
    if title:
        g.fig.suptitle(title)


ddmc_targets = ["AXL", "ABL1", "ABL2", "SRC", "LCK", "LYN", "FRK", "YES1", "HCK", "YAP1", "BLK", "NEK6", "NEK7", "PLK1", "CLK2", "CSNK2A1", "MAPK3", "MAPK1", "BRCA1", "EGFR", "ALK", "INSR"]
bioid_targets = ["AXL", "AHNAK", "FLNA", "SNX1", "ZFYVE16", "TFRC", "DLG5", "CLINT1", "COPG2", "ACSL3", "CTTN", "WWOX", "CTNND1", "TMPO", "EMD", "EGFR", "E41L2", "PLEC", "HSPA9"]

heatmap_ClusterVsTargets_Corr(bioid_targets, omic="RNA", title="")
heatmap_ClusterVsTargets_Corr(bioid_targets, omic="Protein", title="")

# ii = random.sample(range(rnaC.shape[1]), 100)
# targets = list(rnaC.columns[ii])
# heatmap_ClusterVsTargets_Corr(targets, omic="RNA", title="Random genes")

def count_peptides_perCluster(gene, path, ncl, ax):
    """Bar plot of peptide recurrences per cluster"""
    occ = []
    for clN in range(1, ncl + 1):
        cl_genes = list(pd.read_csv(path + str(clN) + ".csv")["Gene"])
        occ.append(cl_genes.count(gene) / len(cl_genes))
    out = pd.DataFrame()
    out["Fraction"] = occ
    out["Cluster"] = np.arange(1, ncl + 1)
    sns.barplot(data=out, x="Cluster", y="Fraction", color="darkblue", edgecolor=".2", ax=ax)
    ax.set_title(gene)

_, ax = plt.subplots(1, 2, figsize=(14, 5))
path = "msresist/data/cluster_members/CPTACmodel_Members_C"
count_peptides_perCluster("AHNAK", path, 24, ax[0])
count_peptides_perCluster("CTNND1", path, 24, ax[1])

_, ax = plt.subplots(1, 4, figsize=(25, 5))
path = "msresist/data/cluster_members/AXLmodel_PAM250_Members_C"
count_peptides_perCluster("AXL", path, 5, ax[0])
count_peptides_perCluster("GAB1", path, 5, ax[1])
count_peptides_perCluster("CTNND1", path, 5, ax[2])
count_peptides_perCluster("MAPK1", path, 5, ax[3])

_, ax = plt.subplots(1, 4, figsize=(25, 5))
path = "msresist/data/cluster_members/AXLmodel_PAM250_Members_C"
count_peptides_perCluster("FLNB", path, 5, ax[0])
count_peptides_perCluster("FLNA", path, 5, ax[1])
count_peptides_perCluster("AHNAK", path, 5, ax[2])
count_peptides_perCluster("TNS1", path, 5, ax[3])