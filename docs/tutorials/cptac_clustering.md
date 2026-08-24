# Tutorial: clustering the CPTAC lung cancer dataset

This tutorial reproduces the core workflow from the DDMC paper: clustering a
clinical phosphoproteomics dataset with heavy missingness, imputing missing
values from the fit, generating cluster motifs, and predicting the upstream
kinases likely responsible for each cluster.

The dataset is the CPTAC lung squamous cell/adenocarcinoma phosphoproteomics
cohort bundled with the package (`ddmc/data/MS/CPTAC/`), pairing each tumor
sample with its adjacent normal (NAT) sample where available.

!!! note
    Some of the functions used here (`predict_upstream_kinases`, and the
    background phosphosite set used by the `"Binomial"` distance) read data
    files using paths relative to the repository root. Run this tutorial
    with the repository root as your working directory, e.g. from a script
    or notebook launched with `uv run`.

## 1. Load and filter the data

`CPTAC.get_p_signal` returns a DataFrame of phosphorylation signal indexed
by the length-11 peptide sequence (5 residues flanking the phosphosite on
each side, phosphoacceptor lowercased), with one column per sample. It's
built from TMT experiments, so `min_experiments` drops peptides seen in
fewer than that many experiments:

```python
from ddmc.datasets import CPTAC, filter_incomplete_peptides

cptac = CPTAC()
p_signal = cptac.get_p_signal(min_experiments=6)
print(p_signal.shape)  # (peptides, samples)
```

Mass spec data is never fully complete, but `DDMC.fit` can already handle
`NaN`s directly (see [How it works](../index.md#how-it-works)). For this
tutorial we additionally drop peptides that are missing in more than 10% of
samples, both to speed up fitting and to keep the imputation step in
step 3 meaningful:

```python
p_signal = filter_incomplete_peptides(p_signal, sample_presence_ratio=0.9)
print(p_signal.shape)
```

## 2. Fit DDMC

`DDMC` takes the number of clusters (`n_components`) and how strongly to
weight sequence motif similarity relative to data similarity (`seq_weight`).
There's no universally correct choice of either — the
[paper](../index.md) explores this via imputation accuracy (see
`ddmc/figures/figureM2.py`) — but a moderate cluster count and weight work
well as a starting point:

```python
from ddmc.clustering import DDMC

model = DDMC(
    n_components=20,
    seq_weight=100,
    distance_method="Binomial",
    random_state=0,
).fit(p_signal)
```

Once fit, `transform` gives the cluster centers (mean phosphorylation signal
per sample, per cluster) and `labels` gives each peptide's assigned cluster:

```python
centers = model.transform(as_df=True)  # samples x clusters
print(centers.shape)

labels = model.labels()  # one cluster index per peptide, aligned to p_signal.index
print(pd.Series(labels).value_counts().head())
```

## 3. Impute missing values

Because DDMC already estimates a center for every cluster on every sample,
it can fill in a peptide's missing samples using its cluster's center. This
is the imputation approach benchmarked in the paper against
mean/zero/PCA imputation:

```python
imputed = model.impute()
assert imputed.isna().sum().sum() == 0
```

## 4. Build cluster motifs (PSSMs)

`get_pssms` computes a position-specific scoring matrix per cluster,
describing which amino acids are enriched at each position relative to a
background distribution of phosphosites:

```python
cluster_names, pssms = model.get_pssms(PsP_background=True)
print(pssms.shape)  # (n_nonempty_clusters, 20 amino acids, 11 positions)
```

You can visualize a cluster's motif as a sequence logo with the plotting
helper used throughout `ddmc/figures/`:

```python
import matplotlib.pyplot as plt
from ddmc.figures.common import plot_motifs

fig, ax = plt.subplots(figsize=(4, 2))
plot_motifs(pssms[0], ax=ax, titles=f"Cluster {cluster_names[0]}")
fig.savefig("cluster_0_motif.svg")
```

## 5. Predict upstream kinases

Comparing each cluster's PSSM to a library of experimentally derived kinase
specificity profiles (position-specific peptide libraries, PSPLs) suggests
which kinase(s) are most likely responsible for phosphorylating peptides in
that cluster:

```python
kinase_distances = model.predict_upstream_kinases(PsP_background=True)
print(kinase_distances.shape)  # kinases x clusters

# Smaller Frobenius distance = better match; show the top hit per cluster.
print(kinase_distances.idxmin(axis=0))
```

## 6. Relate clusters to a clinical feature

Cluster centers can be used as compact per-patient features. As an example,
compare cluster signal between tumor samples with and without an EGFR
mutation, using the mutation calls bundled with the CPTAC dataset:

```python
import numpy as np

mutations = cptac.get_mutations(["EGFR.mutation.status"])

# Restrict to tumor samples (no ".N" suffix) with a known mutation call.
tumor_cols = [
    c for c in p_signal.columns if not c.endswith(".N") and c in mutations.index
]
egfr_mutant = mutations.loc[tumor_cols, "EGFR.mutation.status"].to_numpy()

centers_tumor = centers.loc[tumor_cols]
from scipy.stats import mannwhitneyu

for cluster in centers_tumor.columns:
    values = centers_tumor[cluster].to_numpy()
    _, pval = mannwhitneyu(values[egfr_mutant], values[~egfr_mutant])
    if pval < 0.05:
        print(f"Cluster {cluster}: p={pval:.3g}")
```

This is the same pattern (Mann-Whitney U tests across clusters, then
multiple-testing correction) used by `get_pvals_across_clusters` in
`ddmc/figures/common.py` and the logistic-regression classifiers in
[`ddmc.logistic_regression`](../reference/logistic_regression.md), which use
cluster centers to predict mutation status, tumor-vs-NAT, and hot/cold
immune infiltration across the whole cohort.

## Next steps

- Try `distance_method="PAM250"` and compare the resulting clusters.
- Sweep `seq_weight` from `0` (pure Gaussian mixture) upward and see how
  cluster motifs sharpen — `ddmc/figures/figureM2.py` and `figureM3.py` do
  this systematically via imputation error.
- See the [kinase-inhibitor tutorial](ebdt_clustering.md) for an example
  with a smaller, fully observed dataset.
