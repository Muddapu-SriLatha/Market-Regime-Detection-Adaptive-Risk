"""Unit tests for chronological time series partitioning and zero future leakage."""
import pandas as pd
import pytest

from regime_lab.validation.partitioner import ChronologicalPartitioner


def test_chronological_partition_boundaries():
    """Verify that splits have non-overlapping, strictly increasing dates."""
    dates = pd.date_range("2000-01-01", "2023-12-31", freq="B")
    df = pd.DataFrame({"close": range(len(dates))}, index=dates)

    partitioner = ChronologicalPartitioner(
        train_end="2015-12-31",
        val_end="2019-12-31",
        test_start="2020-01-01"
    )
    splits = partitioner.partition(df)

    # Verify boundaries
    assert splits.train.data.index.max() <= pd.Timestamp("2015-12-31")
    assert splits.validation.data.index.min() > pd.Timestamp("2015-12-31")
    assert splits.validation.data.index.max() <= pd.Timestamp("2019-12-31")
    assert splits.test.data.index.min() >= pd.Timestamp("2020-01-01")

    # Verify zero overlap
    train_set = set(splits.train.data.index)
    val_set = set(splits.validation.data.index)
    test_set = set(splits.test.data.index)

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0


def test_invalid_partition_order_raises():
    """Verify that inverted or unordered partition thresholds raise ValueError."""
    with pytest.raises(ValueError):
        ChronologicalPartitioner(train_end="2020-01-01", val_end="2015-01-01", test_start="2021-01-01")
