from argparse import Namespace
import os
from os.path import abspath, dirname

import gseapy as gp
from iterativeWGCNA.iterativeWGCNA import IterativeWGCNA
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import mygene
import numpy as np
import pandas as pd
import seaborn as sns

COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
PATH_HERE = "/Users/creixell/Desktop/UCLA/Projects/UO1/Computational/GitHub/resistance-MS/msresist/WGCNA"


def get_modules(genes):
    """Maps each gene to its respective module.

    Args:
        genes (numpy.array): Module membership of each gene.

    Returns:
        modules (pandas.DataFrame): DataFrame mapping each gene to a module.
    """
    modules = pd.DataFrame(genes).T
    modules = modules.drop('iteration', axis=1)
    modules = modules.dropna()

    return modules


def lookup_genes(ensembl_genes):
    """Converts ensembl gene IDs to gene names.

    Args:
        ensembl_genes (list[str]): ensembl gene IDs.

    Returns:
        symbols (list[str]): Translated gene names.
    """
    symbols = []
    mg = mygene.MyGeneInfo()
    queries = mg.querymany(
        ensembl_genes,
        return_all=True,
        scopes='ensembl.gene'
    )

    for query in queries:
        symbol = query.get('symbol')
        if symbol is not None:
            symbols.append(symbol)

    return symbols


def fix_label(label):
    """Splits label over two lines.

    Args:
        label (str): Phrase to be split.

    Returns:
        label (str): Provided label split over two lines.
    """
    half = int(len(label) / 2)

    first = label[:half]
    last = label[half:]
    last = last.replace(' ', '\n', 1)

    return first + last


def main():
    ############################################################################
    # DATA IMPORT
    ############################################################################
    # Note that input_data will need to be a .txt file to work with WGCNA, but
    # it's structured just like a .csv, so you can probably save the .csv of
    # your data as a .txt file and it should work.
    # If you need help with what this file should look like, refer to the
    # example file 'data/test.txt'.
    working_dir = f'{PATH_HERE}/output'  # This is where Python will save output
    input_data = f'{PATH_HERE}/data/AXLmutants_RNAseq_merged_filtered.txt'  # Path to your DNA file
    os.makedirs(working_dir, exist_ok=True)

    data = pd.read_csv(
        input_data,
        index_col=0,
        delimiter='\t'
    )

    ############################################################################
    # MODULE IDENTIFICATION
    ############################################################################
    if 'final-membership.txt' in os.listdir(working_dir):
        modules = pd.read_csv(
            f'{working_dir}/final-membership.txt',
            index_col=0,
            sep='\t'
        )
        modules = modules.dropna()
    else:
        wgcna_args = {
            'inputFile': input_data,
            'workingDir': working_dir,
            'verbose': False,
            'debug': False,
            'wgcnaParameters': {
                'numericLabels': True,
                'networkType': 'signed',
                'minKMEtoStay': 0.8,
                'minCoreKME': 0.8,
                'minModuleSize': 20,
                'reassignThreshold': 0.05,
                'power': 6,
                'saveTOMs': False
            },
            'enableWGCNAThreads': True,
            'skipSaveBlocks': False,
            'finalMergeCutHeight': 0.05,
            'gzipTOMs': False
        }
        ns = Namespace(**wgcna_args)
        wgcna = IterativeWGCNA(ns)
        wgcna.run()
        modules = get_modules(wgcna.genes.genes)
        modules = modules.rename(columns={'module': 'Module'})
        modules.to_csv("msresist/WGCNA/output/RNAseq_Modules", sep="\t")

    ############################################################################
    # MODULE PLOTTING
    ############################################################################
    names = sorted(list(set(modules.loc[:, 'Module'])))
    module_expression = pd.DataFrame(
        index=names,
        columns=data.columns
    )
    for name in names:
        in_module = modules.loc[modules.loc[:, 'Module'] == name]
        module = data.loc[in_module.index, :]
        module_expression.loc[name, :] = module.mean()

    plt.figure(figsize=(6, 6))
    bound = module_expression.abs().max().max()
    sns.heatmap(
        module_expression.astype(float),
        center=0,
        cmap='vlag',
        vmax=bound,
        vmin=-bound,
        cbar_kws={
            'label': 'Mean Module Expression'
        }
    )
    plt.yticks(
        np.arange(
            0.5,
            module_expression.shape[0]
        ),
        module_expression.index,
        rotation=0
    )
    plt.ylabel('Module')
    plt.xlabel('Component')
    plt.subplots_adjust(
        left=0.2,
        bottom=0.05,
        right=0.9,
        top=0.95
    )
    plt.savefig(f'{working_dir}/modules_v_components.png')

    ############################################################################
    # ENRICHMENT ANALYSIS
    ############################################################################
    # The below list picks the gene sets that are run as part of the enrichment.
    # These provide a decent general search, but you can pick more specific
    # sets from the options here: https://maayanlab.cloud/Enrichr/#libraries
    gene_sets = [
        'GO_Biological_Process_2021',
        'GO_Cellular_Component_2021',
        'GO_Molecular_Function_2021'
    ]
    n_cols = len(gene_sets)
    for name in names:
        # The code below performs the enrichment analysis and saves the result
        # to a .csv in the working_dir
        module = modules.loc[modules.loc[:, 'Module'] == name]
        ensembl = list(module.index)

        # The step below may not be necessary for you; our genes were labeled
        # with their ensembl IDs (e.g. ENSG00000004779) and need to be converted
        # to names that Enrichr recognizes first
        genes = lookup_genes(ensembl)

        result = gp.enrichr(
            list(genes),
            gene_sets=gene_sets,
            organism='Human'
        ).results
        result = result.set_index('Term', drop=True)
        result.to_csv(f'{working_dir}/{name}_GO.csv')

        # From here on, the rest of this loop plots the gene sets present in
        # each module; all the enrichment analyses are done above!
        # The below will save the enrichment analysis results for each module
        # as .pngs in the working_dir.
        fig, axs = plt.subplots(
            figsize=(12, 8),
            ncols=n_cols,
            nrows=2,
            constrained_layout=True
        )
        cols = range(n_cols)
        for col, gene_set in zip(cols, gene_sets):
            ax_p = axs[0, col]
            ax_combined = axs[1, col]

            combined_score = result.loc[
                result['Gene_set'] == gene_set,
                'Combined Score'
            ]
            combined_score = combined_score.sort_values(ascending=False)
            combined_score = combined_score.iloc[:10]

            p_value = result.loc[
                result['Gene_set'] == gene_set,
                'Adjusted P-value'
            ]
            p_value = p_value.sort_values(ascending=True)
            p_value = p_value.iloc[:10]

            ax_p.barh(
                range(len(p_value)),
                p_value,
                color=COLOR_CYCLE[1]
            )
            ax_p.plot(
                [0.05, 0.05],
                [-1, 100],
                alpha=0.25,
                color='k',
                linestyle='--'
            )
            ax_combined.barh(
                range(len(combined_score)),
                combined_score,
                color=COLOR_CYCLE[1]
            )

            ax_p.set_xscale('log')
            ax_p.set_xlim(right=10)
            ax_p.set_ylim([-1, 10])
            ax_p.set_yticks([])
            ax_combined.set_yticks([])
            ax_combined.set_ylim([-1, 10])

            transform_p = transforms.blended_transform_factory(
                ax_p.transAxes,
                ax_p.transData
            )
            transform_combined = transforms.blended_transform_factory(
                ax_combined.transAxes,
                ax_combined.transData
            )

            for i in range(len(p_value)):
                p_label = p_value.index[i]
                combined_label = combined_score.index[i]
                if len(p_label) > 60:
                    p_label = fix_label(p_label)
                if len(combined_label) > 60:
                    combined_label = fix_label(combined_label)

                ax_p.text(
                    0.025,
                    i,
                    p_label,
                    fontsize=6,
                    va='center',
                    ha='left',
                    transform=transform_p
                )
                ax_combined.text(
                    0.025,
                    i,
                    combined_label,
                    fontsize=6,
                    va='center',
                    ha='left',
                    transform=transform_combined
                )

            ax_combined.set_ylabel(gene_set.replace("_", " "))
            ax_p.set_ylabel(gene_set.replace("_", " "))
            ax_combined.set_xlabel('Combined Score')
            ax_p.set_xlabel('Fisher Exact Test P-value')

        fig.suptitle(name)
        fig.savefig(f'{working_dir}/{name}_GO.png')


if __name__ == '__main__':
    main()
