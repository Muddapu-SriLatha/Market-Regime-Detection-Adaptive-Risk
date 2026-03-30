"""Validation, regime breakdown analytics, and robustness testing subpackage."""
from regime_lab.validation.partitioner import ChronologicalPartitioner, DataSplits
from regime_lab.validation.regime_analysis import analyze_performance_by_regime, RegimePerformanceSummary
from regime_lab.validation.sensitivity import run_sensitivity_experiments, SensitivityResults

__all__ = [
    "ChronologicalPartitioner",
    "DataSplits",
    "analyze_performance_by_regime",
    "RegimePerformanceSummary",
    "run_sensitivity_experiments",
    "SensitivityResults"
]
