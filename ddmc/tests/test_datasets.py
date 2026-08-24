import numpy as np
import pandas as pd
import pytest

from ddmc.datasets import CPTAC, filter_incomplete_peptides, select_peptide_subset


def test_filter_incomplete_peptides_by_sample_presence_ratio():
    p_signal = pd.DataFrame(
        {
            "s1": [1.0, 1.0, 1.0],
            "s2": [1.0, 1.0, np.nan],
            "s3": [1.0, np.nan, np.nan],
        },
        index=["complete", "two_thirds", "one_third"],
    )

    filtered = filter_incomplete_peptides(p_signal, sample_presence_ratio=0.5)

    assert list(filtered.index) == ["complete", "two_thirds"]


def test_filter_incomplete_peptides_by_min_experiments():
    p_signal = pd.DataFrame(
        {
            "s1": [1.0, np.nan],
            "s2": [1.0, 1.0],
            "s3": [np.nan, np.nan],
            "s4": [np.nan, 1.0],
        },
        index=["seen_in_one_experiment", "seen_in_both_experiments"],
    )
    sample_to_experiment = np.array(["e1", "e1", "e2", "e2"])

    filtered = filter_incomplete_peptides(
        p_signal, min_experiments=2, sample_to_experiment=sample_to_experiment
    )

    assert list(filtered.index) == ["seen_in_both_experiments"]


def test_filter_incomplete_peptides_requires_a_threshold():
    p_signal = pd.DataFrame({"s1": [1.0]}, index=["a"])
    with pytest.raises(ValueError, match="Must specify"):
        filter_incomplete_peptides(p_signal)


def test_select_peptide_subset_by_keep_num():
    p_signal = pd.DataFrame({"s1": range(10)}, index=[f"p{i}" for i in range(10)])
    subset = select_peptide_subset(p_signal, keep_num=4)
    assert subset.shape[0] == 4


def test_select_peptide_subset_by_keep_ratio():
    p_signal = pd.DataFrame({"s1": range(10)}, index=[f"p{i}" for i in range(10)])
    subset = select_peptide_subset(p_signal, keep_ratio=0.3)
    assert subset.shape[0] == 3


def test_get_patients_with_nat_and_tumor():
    samples = [
        "C3L.00001",
        "C3L.00001.N",
        "C3L.00002",  # tumor only, no matching NAT
        "C3L.00003.N",  # NAT only, no matching tumor
        "Tumor.Only.IR",  # pooled reference channel, should be ignored
    ]

    patients = CPTAC().get_patients_with_nat_and_tumor(samples)

    assert list(patients) == ["C3L.00001"]


def test_get_tumor_or_nat():
    samples = ["C3L.00001", "C3L.00001.N", "C3L.00002"]
    result = CPTAC().get_tumor_or_nat(samples)
    np.testing.assert_array_equal(result, [True, False, True])
