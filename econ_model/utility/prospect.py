from typing import Dict, Any, Optional

import numpy as np

from .base import UtilityModel


class ProspectUtility(UtilityModel):
    """Prospect Theory utility with value function and probability weighting.

    This is a pragmatic implementation for our binary decision (work vs not work).
    We construct a scalar "outcome index" as a linear form X @ beta_outcome + c,
    then pass it through a PT value function v(·). The decision layer (logit)
    interprets v(·) as systematic component of utility.

    Value function (Tversky & Kahneman 1992):
      v(x) = x^alpha          if x >= 0
           = -lambda * (-x)^beta  if x < 0

    Probability weighting (Prelec 1998) is not explicitly used in binary logit
    because choice probability is given by logit; however, we expose parameters
    and keep the structure for extension to multi-outcome contexts.
    """

    def __init__(self, n_features: Optional[int] = None,
                 alpha: float = 0.88, beta: float = 0.88, lam: float = 2.25,
                 pw_gamma: float = 0.65):
        self.beta_vec: Optional[np.ndarray] = None  # outcome index weights
        self.intercept: float = 0.0
        self.n_features = n_features
        # PT parameters
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lam = float(lam)
        self.pw_gamma = float(pw_gamma)

    def _value_function(self, x: np.ndarray) -> np.ndarray:
        pos = x >= 0
        out = np.empty_like(x)
        out[pos] = np.power(x[pos], self.alpha)
        out[~pos] = -self.lam * np.power(-x[~pos], self.beta)
        return out

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.beta_vec is None:
            if self.n_features is None:
                self.n_features = X.shape[1]
            self.beta_vec = np.zeros(self.n_features, dtype=float)
        idx = X @ self.beta_vec + self.intercept
        return self._value_function(idx)

    def get_params(self) -> Dict[str, Any]:
        return {
            "beta": None if self.beta_vec is None else self.beta_vec.tolist(),
            "intercept": float(self.intercept),
            "alpha": self.alpha,
            "beta_v": self.beta,
            "lambda": self.lam,
            "pw_gamma": self.pw_gamma,
        }

    def set_params(self, **kwargs: Any) -> None:
        beta = kwargs.get("beta")
        if beta is not None:
            self.beta_vec = np.asarray(beta, dtype=float)
            self.n_features = self.beta_vec.shape[0]
        if "intercept" in kwargs:
            self.intercept = float(kwargs["intercept"])
        # PT parameters (optional tuning)
        if "alpha" in kwargs:
            self.alpha = float(kwargs["alpha"])
        if "beta_v" in kwargs:
            self.beta = float(kwargs["beta_v"])
        if "lambda" in kwargs:
            self.lam = float(kwargs["lambda"])
        if "pw_gamma" in kwargs:
            self.pw_gamma = float(kwargs["pw_gamma"])

    def gradient_wrt_beta(self, X: np.ndarray) -> np.ndarray:
        # dv/dx * d(x)/d(beta) = v'(idx) * X
        if self.beta_vec is None:
            self.beta_vec = np.zeros(X.shape[1], dtype=float)
        idx = X @ self.beta_vec + self.intercept
        pos = idx >= 0
        vprime = np.empty_like(idx)
        # derivative of x^alpha = alpha * x^(alpha-1) for x>0
        vprime[pos] = self.alpha * np.power(np.maximum(idx[pos], 1e-12), self.alpha - 1.0)
        # derivative of -lambda * (-x)^beta = -lambda * beta * (-x)^(beta-1) * (-1)
        # = lambda * beta * (-x)^(beta-1)
        vprime[~pos] = self.lam * self.beta * np.power(np.maximum(-idx[~pos], 1e-12), self.beta - 1.0)
        return (vprime[:, None] * X)

    def gradient_wrt_intercept(self, X: np.ndarray) -> np.ndarray:
        if self.beta_vec is None:
            self.beta_vec = np.zeros(X.shape[1], dtype=float)
        idx = X @ self.beta_vec + self.intercept
        pos = idx >= 0
        vprime = np.empty_like(idx)
        vprime[pos] = self.alpha * np.power(np.maximum(idx[pos], 1e-12), self.alpha - 1.0)
        vprime[~pos] = self.lam * self.beta * np.power(np.maximum(-idx[~pos], 1e-12), self.beta - 1.0)
        return vprime


