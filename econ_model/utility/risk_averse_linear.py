from typing import Dict, Any, Optional

import numpy as np

from .base import UtilityModel


class RiskAverseLinear(UtilityModel):
    """风险厌恶的二次型效用：U = Xβ + c - 0.5 * γ * (Xβ)^2。

    γ>=0 控制风险厌恶强度；对二元决策可模拟边际效用递减。
    """

    def __init__(self, n_features: Optional[int] = None, gamma: float = 0.1):
        self.beta: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.n_features = n_features
        self.gamma = float(gamma)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            if self.n_features is None:
                self.n_features = X.shape[1]
            self.beta = np.zeros(self.n_features, dtype=float)
        idx = X @ self.beta + self.intercept
        return idx - 0.5 * self.gamma * (idx ** 2)

    def get_params(self) -> Dict[str, Any]:
        return {"beta": None if self.beta is None else self.beta.tolist(), "intercept": float(self.intercept), "gamma": self.gamma}

    def set_params(self, **kwargs: Any) -> None:
        beta = kwargs.get("beta")
        if beta is not None:
            self.beta = np.asarray(beta, dtype=float)
            self.n_features = self.beta.shape[0]
        if "intercept" in kwargs:
            self.intercept = float(kwargs["intercept"])
        if "gamma" in kwargs:
            self.gamma = float(kwargs["gamma"])

    def gradient_wrt_beta(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            self.beta = np.zeros(X.shape[1], dtype=float)
        idx = X @ self.beta + self.intercept
        # dU/dβ = (1 - γ*idx) * X
        return ((1.0 - self.gamma * idx)[:, None] * X)

    def gradient_wrt_intercept(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            self.beta = np.zeros(X.shape[1], dtype=float)
        idx = X @ self.beta + self.intercept
        # dU/dc = 1 - γ*idx
        return (1.0 - self.gamma * idx)


