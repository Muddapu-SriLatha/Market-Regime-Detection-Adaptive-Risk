"""
Gaussian Mixture Model Regime Detector
=======================================
Statistical regime detection using scikit-learn Gaussian Mixture Models (GMM).
Features:
- Scaler and GMM parameters fitted strictly on in-sample training data
- Explicit resolution of unsupervised label ambiguity by ordering regimes by training volatility:
    Regime 0: Low Risk (calm, lowest volatility)
    Regime 1: Medium Risk (moderate volatility, transitional)
    Regime 2: High Risk (highest volatility, turbulence/crisis)
- Posterior probability extraction for regime uncertainty diagnostics
- Causal forward prediction for out-of-sample evaluation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeModelSummary:
    """Diagnostic profile of fitted regimes."""
    n_regimes: int
    regime_names: Dict[int, str]
    cluster_means: pd.DataFrame
    training_frequencies: Dict[int, float]
    mean_volatility_by_regime: Dict[int, float]
    mean_momentum_by_regime: Dict[int, float]
    aic: float
    bic: float


class GMMRegimeDetector:
    """
    Gaussian Mixture Model detector for unsupervised market regime identification.
    Guarantees strict separation of training parameters and out-of-sample forward inference.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
        max_iter: int = 200
    ):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.max_iter = max_iter

        self.scaler: Optional[StandardScaler] = None
        self.gmm: Optional[GaussianMixture] = None
        self.feature_names: List[str] = []
        self.cluster_to_regime: Dict[int, int] = {}
        self.is_fitted: bool = False
        self.summary_: Optional[RegimeModelSummary] = None

    def fit(self, train_features: pd.DataFrame) -> "GMMRegimeDetector":
        """
        Fit feature scaler and GMM parameters strictly on training features.

        Parameters
        ----------
        train_features : pd.DataFrame
            Matrix of feature values for the training period. Any rows with NaNs
            (e.g. from lookback warmups) are dropped prior to fitting.
        """
        clean_df = train_features.dropna()
        if len(clean_df) < self.n_regimes * 20:
            raise ValueError(
                f"Insufficient training observations ({len(clean_df)}) to fit {self.n_regimes} regimes."
            )

        self.feature_names = list(clean_df.columns)
        if "short_vol" not in self.feature_names:
            raise ValueError("Training features must include 'short_vol' for economic regime ordering.")

        # 1. Fit scaler strictly on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(clean_df[self.feature_names].values)

        # 2. Fit Gaussian Mixture Model
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        raw_clusters = self.gmm.fit_predict(X_scaled)

        # 3. Resolve unsupervised label ambiguity:
        # Order clusters by mean realized volatility on the training set
        vol_col_idx = self.feature_names.index("short_vol")
        raw_vol_series = clean_df["short_vol"].values
        cluster_vols = {}
        for c in range(self.n_regimes):
            mask = (raw_clusters == c)
            cluster_vols[c] = float(np.mean(raw_vol_series[mask])) if np.any(mask) else 0.0

        # Sort cluster indices by ascending volatility
        sorted_clusters = sorted(cluster_vols.keys(), key=lambda c: cluster_vols[c])
        # Mapping: raw_cluster -> ordered_regime (0 = Low Risk, 1 = Med Risk, 2 = High Risk)
        self.cluster_to_regime = {raw_c: ordered_r for ordered_r, raw_c in enumerate(sorted_clusters)}

        ordered_labels = np.array([self.cluster_to_regime[c] for c in raw_clusters])

        # 4. Compile diagnostic summary
        regime_names = {
            0: "Low Risk (Calm Bull)",
            1: "Medium Risk (Transitional)",
            2: "High Risk (Turbulent Crisis)"
        }
        train_freqs = {r: float(np.mean(ordered_labels == r)) for r in range(self.n_regimes)}
        mean_vols = {r: float(np.mean(raw_vol_series[ordered_labels == r])) for r in range(self.n_regimes)}
        
        has_mom = "norm_momentum" in self.feature_names
        raw_mom = clean_df["norm_momentum"].values if has_mom else np.zeros(len(clean_df))
        mean_moms = {r: float(np.mean(raw_mom[ordered_labels == r])) for r in range(self.n_regimes)}

        # Means table
        means_data = []
        for r in range(self.n_regimes):
            raw_c = sorted_clusters[r]
            scaled_mean = self.gmm.means_[raw_c]
            orig_mean = self.scaler.inverse_transform(scaled_mean.reshape(1, -1))[0]
            means_data.append(orig_mean)
        means_df = pd.DataFrame(means_data, columns=self.feature_names, index=[regime_names[r] for r in range(self.n_regimes)])

        self.summary_ = RegimeModelSummary(
            n_regimes=self.n_regimes,
            regime_names=regime_names,
            cluster_means=means_df,
            training_frequencies=train_freqs,
            mean_volatility_by_regime=mean_vols,
            mean_momentum_by_regime=mean_moms,
            aic=float(self.gmm.aic(X_scaled)),
            bic=float(self.gmm.bic(X_scaled))
        )

        self.is_fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Assign regime labels (0, 1, 2) to feature observations.
        Preserves original DatetimeIndex. NaNs are retained as NaN (or -1).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted on training data before calling predict.")

        clean_mask = ~features[self.feature_names].isna().any(axis=1)
        labels = pd.Series(index=features.index, dtype=float)
        labels[:] = np.nan

        if not clean_mask.any():
            return labels

        X = features.loc[clean_mask, self.feature_names].values
        X_scaled = self.scaler.transform(X)
        raw_preds = self.gmm.predict(X_scaled)
        ordered_preds = np.array([self.cluster_to_regime[c] for c in raw_preds], dtype=float)
        labels.loc[clean_mask] = ordered_preds

        return labels

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Compute posterior regime probabilities P(Regime k | X_t).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted on training data before calling predict_proba.")

        clean_mask = ~features[self.feature_names].isna().any(axis=1)
        proba_df = pd.DataFrame(
            index=features.index,
            columns=[f"P_regime_{r}" for r in range(self.n_regimes)],
            dtype=float
        )
        proba_df[:] = np.nan

        if not clean_mask.any():
            return proba_df

        X = features.loc[clean_mask, self.feature_names].values
        X_scaled = self.scaler.transform(X)
        raw_proba = self.gmm.predict_proba(X_scaled)

        # Map probabilities from raw cluster to ordered regime
        ordered_proba = np.zeros_like(raw_proba)
        for raw_c, ord_r in self.cluster_to_regime.items():
            ordered_proba[:, ord_r] = raw_proba[:, raw_c]

        proba_df.loc[clean_mask] = ordered_proba
        return proba_df
