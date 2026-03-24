"""Strategy subpackage for fair fixed vs adaptive risk comparison."""
from regime_lab.strategy.signals import compute_trend_signal
from regime_lab.strategy.fixed_risk import FixedRiskStrategy
from regime_lab.strategy.adaptive_risk import AdaptiveRiskStrategy

__all__ = [
    "compute_trend_signal",
    "FixedRiskStrategy",
    "AdaptiveRiskStrategy"
]
