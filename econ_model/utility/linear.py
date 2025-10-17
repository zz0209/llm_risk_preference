from typing import Dict, Any, Optional

import numpy as np

from .base import UtilityModel


class LinearUtility(UtilityModel):
    """线性可加效用 U = X @ beta + intercept。

    - 仅做前向；参数拟合由决策模型（如 Logit）完成。
    - 兼容未来扩展：可以在外部构造非线性特征再输入本模型。
    """

    def __init__(self, n_features: Optional[int] = None):
        self.beta: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.n_features = n_features

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            if self.n_features is None:
                self.n_features = X.shape[1]
            self.beta = np.zeros(self.n_features, dtype=float)
        return X @ self.beta + self.intercept

    def get_params(self) -> Dict[str, Any]:
        return {
            "beta": None if self.beta is None else self.beta.tolist(),
            "intercept": float(self.intercept),
        }

    def set_params(self, **kwargs: Any) -> None:
        beta = kwargs.get("beta")
        if beta is not None:
            self.beta = np.asarray(beta, dtype=float)
            self.n_features = self.beta.shape[0]
        if "intercept" in kwargs:
            self.intercept = float(kwargs["intercept"])


