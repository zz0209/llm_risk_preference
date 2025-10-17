from typing import Dict, Any

import numpy as np


class OLS:
    """最小二乘（闭式解），用于工时的基线回归。
    y_hat = X @ w + b
    """

    def __init__(self):
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n, d = X.shape
        X_aug = np.hstack([X, np.ones((n, 1))])
        # 岭回归闭式解（提高数值稳定性）
        reg = 1e-2 * np.eye(d + 1)
        try:
            w = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(X_aug.T @ X_aug + reg) @ (X_aug.T @ y)
        self.coef_ = w[:d]
        self.intercept_ = float(w[d])
        y_hat = X @ self.coef_ + self.intercept_
        resid = y - y_hat
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        return {"rmse": rmse, "mae": mae}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


