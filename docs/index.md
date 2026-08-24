# DDMC: dual data and motif clustering

DDMC clusters phosphoproteomic mass spectrometry data by jointly considering
two signals for each phosphopeptide:

1. **Data similarity** — how similar its phosphorylation signal is across
   samples/conditions to other peptides (the same objective a standard
   Gaussian mixture model clusters on).
2. **Motif similarity** — how similar the amino acid sequence surrounding the
   phosphosite is to other peptides in the same cluster, which acts as a
   proxy for being regulated by the same upstream kinase.

Combining both lets DDMC produce clusters that are both more robust to the
heavy missingness typical of mass-spec data and more directly interpretable
in terms of upstream kinase biology, compared to clustering on either signal
alone.

DDMC is described in:

> Creixell M, Meyer AS. [Dual data and motif clustering improves the modeling
> and interpretation of phosphoproteomic
> data](https://pubmed.ncbi.nlm.nih.gov/35360705/). *Cell Rep Methods*. 2022
> Feb 28;2(2):100167. doi:
> [10.1016/j.crmeth.2022.100167](https://doi.org/10.1016/j.crmeth.2022.100167)

**Abstract:** Cell signaling is orchestrated in part through a network of
protein kinases and phosphatases. Dysregulation of kinase signaling is
widespread in diseases such as cancer and is readily targetable through
inhibitors. Mass spectrometry-based analysis can provide a global view of
kinase regulation, but mining these data is complicated by its stochastic
coverage of the proteome, measurement of substrates rather than kinases, and
the scale of the data. Here, we implement a dual data and motif clustering
(DDMC) strategy that simultaneously clusters peptides into similarly
regulated groups based on their variation and their sequence profile. We
show that this can help to identify putative upstream kinases and supply
more robust clustering. We apply this clustering to clinical proteomic
profiling of lung cancer and identify conserved proteomic signatures of
tumorigenicity, genetic mutations, and immune infiltration. We propose that
DDMC provides a general and flexible clustering strategy for the analysis of
phosphoproteomic data.

## How it works

`DDMC` (in [`ddmc.clustering`][ddmc.clustering]) subclasses scikit-learn's
`sklearn.mixture.GaussianMixture` and runs the same
expectation-maximization algorithm, with two changes:

- In the **E step**, the log-probability of each peptide belonging to each
  cluster is the usual Gaussian mixture log-probability *plus* a sequence
  term: `seq_weight * seq_dist.logWeights`, where `seq_dist` scores how well
  a peptide's sequence matches each cluster's current motif.
- In the **M step**, in addition to the normal Gaussian mixture update
  (recomputing each cluster's mean and variance across samples), the
  per-cluster sequence motif is refit from the current soft cluster
  assignments (responsibilities).

Two sequence-distance methods are available (`distance_method=`):

- `"Binomial"` (default) — for each cluster, scores how enriched each
  amino acid is at each position relative to a background phosphosite
  distribution, using the binomial approach of
  [Schwartz & Gygi, *Nat Biotechnol* 2005](https://doi.org/10.1038/nbt1146).
  See [`ddmc.binomial.Binomial`][ddmc.binomial.Binomial].
- `"PAM250"` — scores sequences by average PAM250 substitution-matrix
  similarity to the other sequences currently assigned to a cluster. See
  [`ddmc.pam250.PAM250`][ddmc.pam250.PAM250].

The `seq_weight` argument controls the relative contribution of the sequence
term: `seq_weight=0` reduces `DDMC` to an ordinary Gaussian mixture model
over the data alone (this is a useful sanity check — see
`ddmc/tests/test_cluster.py`), while larger values weight the motif more
heavily. Missing values (common in mass-spec data due to its stochastic
proteome coverage) are handled by imputing them from the current cluster
centers between EM iterations (via `SoftImpute`), so `DDMC.fit` accepts data
with `NaN`s directly.

Once fit, a model can be used to:

- get cluster centers across samples (`transform`) and per-peptide cluster
  assignments (`labels`)
- fill in an imputed version of the input data (`impute`)
- build a position-specific scoring matrix (PSSM) per cluster (`get_pssms`)
- compare cluster PSSMs against a library of kinase specificity profiles to
  predict likely upstream kinases per cluster (`predict_upstream_kinases`)

## Installation

DDMC targets Python 3.12+ and is managed with [uv](https://docs.astral.sh/uv/).
Clone the repository and install the project along with its dependencies:

```sh
git clone https://github.com/meyer-lab/DDMC.git
cd DDMC
uv sync
```

Run the test suite with:

```sh
make test
```

## Where to go next

- Read the [CPTAC lung cancer tutorial](tutorials/cptac_clustering.md) to see
  DDMC applied to the clinical proteomics dataset used in the paper,
  including handling missing values and predicting upstream kinases.
- Read the [kinase-inhibitor tutorial](tutorials/ebdt_clustering.md) to see
  DDMC applied to a small, complete (no missing values) drug-perturbation
  dataset.
- Browse the [API reference](reference/clustering.md) for details on every
  public function and class.
