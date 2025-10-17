from typing import Dict, Any

import numpy as np
from sklearn.linear_model import HuberRegressor


class Huber:
    """Huber 回归（鲁棒），用于工时的对比模型。

    使用 sklearn 的 HuberRegressor，默认 epsilon=1.35，alpha=1e-4。
    """

    def __init__(self, epsilon: float = 1.35, alpha: float = 1e-4):
        self.model = HuberRegressor(epsilon=epsilon, alpha=alpha, fit_intercept=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        self.model.fit(X, y)
        y_hat = self.model.predict(X)
        resid = y - y_hat
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        return {"rmse": rmse, "mae": mae}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_

    @property
    def intercept_(self) -> float:
        return float(self.model.intercept_)


