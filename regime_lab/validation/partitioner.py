"""
Chronological Time Series Partitioner
======================================
Partitions time-series data into strict, non-overlapping chronological splits:
1. In-Sample Training (2000-01-01 to 2015-12-31)
2. Validation / Tuning (2016-01-01 to 2019-12-31)
3. Untouched Out-of-Sample Holdout (2020-01-01 to present)

Guarantees zero future look-ahead leakage across partition boundaries.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd


@dataclass
class SplitWindow:
    """Descriptor for a single chronological slice."""
    name: str
    start_date: str
    end_date: str
    total_bars: int
    data: pd.DataFrame


@dataclass
class DataSplits:
    """Encapsulates all 3 chronological partitions."""
    train: SplitWindow
    validation: SplitWindow
    test: SplitWindow

    def summary(self) -> pd.DataFrame:
        rows = [
            (s.name, s.start_date, s.end_date, s.total_bars)
            for s in [self.train, self.validation, self.test]
        ]
        return pd.DataFrame(rows, columns=["Partition", "Start Date", "End Date", "Total Bars"])


class ChronologicalPartitioner:
    """
    Partitions market data without random shuffling, preserving temporal ordering.
    """

    def __init__(
        self,
        train_end: str = "2015-12-31",
        val_end: str = "2019-12-31",
        test_start: str = "2020-01-01"
    ):
        self.train_end = pd.Timestamp(train_end)
        self.val_end = pd.Timestamp(val_end)
        self.test_start = pd.Timestamp(test_start)

        if not (self.train_end < self.val_end < self.test_start):
            raise ValueError(
                f"Invalid partition ordering: train_end ({train_end}) < val_end ({val_end}) < test_start ({test_start})"
            )

    def partition(self, df: pd.DataFrame) -> DataSplits:
        """
        Slice dataframe into Train, Validation, and Test sets based on index dates.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Dataframe index must be a DatetimeIndex.")

        train_mask = df.index <= self.train_end
        val_mask = (df.index > self.train_end) & (df.index <= self.val_end)
        test_mask = df.index >= self.test_start

        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]
        test_df = df.loc[test_mask]

        if len(train_df) == 0:
            raise ValueError("Training partition is empty.")
        if len(val_df) == 0:
            raise ValueError("Validation partition is empty.")
        if len(test_df) == 0:
            raise ValueError("Test partition is empty.")

        train_window = SplitWindow(
            name="In-Sample Train",
            start_date=str(train_df.index[0].date()),
            end_date=str(train_df.index[-1].date()),
            total_bars=len(train_df),
            data=train_df
        )

        val_window = SplitWindow(
            name="Validation",
            start_date=str(val_df.index[0].date()),
            end_date=str(val_df.index[-1].date()),
            total_bars=len(val_df),
            data=val_df
        )

        test_window = SplitWindow(
            name="Out-of-Sample Test",
            start_date=str(test_df.index[0].date()),
            end_date=str(test_df.index[-1].date()),
            total_bars=len(test_df),
            data=test_df
        )

        return DataSplits(
            train=train_window,
            validation=val_window,
            test=test_window
        )
