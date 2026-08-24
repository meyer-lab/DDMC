# Tutorial: clustering a kinase-inhibitor perturbation dataset

The CPTAC tutorial works with a large, heavily-missing clinical dataset. This
tutorial instead uses the small, fully observed MCF7 kinase-inhibitor
dataset from [Hijazi et al., *Nat Biotechnol* 2020](https://www.nature.com/articles/s41587-019-0391-9)
(bundled as `ddmc/data/Validations/Computational/ebdt_mcf7.csv`), which is
convenient for quickly comparing clustering settings since there's no
missing-value handling to think about.

Each column is the fold-change in phosphorylation signal for the MCF7 breast
cancer cell line treated with a given kinase inhibitor, relative to control;
each row is a phosphopeptide.

## 1. Load the data

`EBDT.get_p_signal` maps the dataset's peptide identifiers onto the human
proteome to build the same length-11 sequence representation
(`ddmc/motifs.py`) that `CPTAC.get_p_signal` uses, so the resulting
DataFrame is a drop-in match for `DDMC.fit`:

```python
from ddmc.datasets import EBDT

p_signal = EBDT().get_p_signal()
print(p_signal.shape)  # (peptides, inhibitors)
print(p_signal.isna().sum().sum())  # 0 — no missing values in this dataset
```

## 2. Fit with both distance methods

Because this dataset is small, it's cheap to compare DDMC's two sequence
distance methods directly:

```python
from ddmc.clustering import DDMC

model_binomial = DDMC(
    n_components=15, seq_weight=50, distance_method="Binomial", random_state=0
).fit(p_signal)

model_pam250 = DDMC(
    n_components=15, seq_weight=50, distance_method="PAM250", random_state=0
).fit(p_signal)

print(model_binomial.score(), model_pam250.score())
```

`seq_weight=0` disables the sequence term entirely, making `DDMC` equivalent
to a plain `sklearn.mixture.GaussianMixture` fit on `p_signal.values` — a
useful sanity check when tuning `seq_weight` upward from zero (this is
exactly what `ddmc/tests/test_cluster.py::test_wins` checks).

## 3. Inspect which inhibitors distinguish each cluster

Since columns here are inhibitors rather than patient samples, cluster
centers directly show which inhibitors most shift each cluster's
phosphorylation:

```python
centers = model_binomial.transform(as_df=True)  # inhibitors x clusters

# Inhibitors with the strongest (most negative/positive) effect on cluster 0
print(centers[0].sort_values().head())
print(centers[0].sort_values().tail())
```

For example, clusters most suppressed by PI3K/AKT inhibitors (columns like
`MCF7.GDC0941.fold`, `MCF7.MK2206.fold`) are candidates for being downstream
of PI3K/AKT signaling — which `predict_upstream_kinases` (see the
[CPTAC tutorial](cptac_clustering.md#5-predict-upstream-kinases)) can help
confirm from the sequence motif side, independent of the perturbation data.

## 4. Compare cluster assignments between the two distance methods

Since both models were fit with the same `n_components` and `random_state`,
you can directly compare how much sequence information changes cluster
membership:

```python
import numpy as np

agreement = np.mean(model_binomial.labels() == model_pam250.labels())
print(f"Fraction of peptides with the same cluster label: {agreement:.2f}")
```

Exact label agreement isn't expected to be high — cluster *indices* aren't
aligned between independent fits — but comparing the resulting motifs
(`get_pssms`) or projecting both sets of centers with
`plot_pca_on_cluster_centers` from `ddmc/figures/common.py` is a more
meaningful way to compare the two distance methods' structure.

## Next steps

- Increase `seq_weight` and watch clusters become more homogeneous in
  sequence motif at the cost of separation in the inhibitor-response data —
  the same tradeoff explored on CPTAC data in `ddmc/figures/figureM2.py`.
- Combine this with `predict_upstream_kinases` to check whether a cluster's
  predicted kinase matches the inhibitor(s) that most affect it.
