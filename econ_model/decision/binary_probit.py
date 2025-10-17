from typing import Dict, Any, Optional

import numpy as np
from scipy.stats import norm

from econ_model.utility import UtilityModel


class BinaryProbit:
    """二元 Probit，使用GLM-IRLS近似（工作向量 + 权重）并兼容任意 UtilityModel。

    链接：Φ^{-1}(μ)=η，其中 μ=Φ(η)，η=U(X)。
    IRLS 权重：w = (dμ/dη)^2 / (μ(1-μ))，其中 dμ/dη = φ(η)。
    工作响应：z_work = η + (y-μ)/(dμ/dη)。
    使用 J = [∂η/∂β, ∂η/∂intercept] 进行参数更新 δθ = (J^T W J + λI)^{-1} J^T W (z_work - η)。
    """

    def __init__(self, utility: UtilityModel, l2: float = 1e-3, max_iter: int = 100, tol: float = 1e-6):
        self.utility = utility
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n, d = X.shape
        params = self.utility.get_params()
        beta = np.zeros(d) if params.get("beta") is None else np.asarray(params["beta"], dtype=float)
        intercept = float(params.get("intercept", 0.0))

        for it in range(self.max_iter):
            # forward
            self.utility.set_params(beta=beta, intercept=intercept)
            eta = self.utility.evaluate(X)
            # Clamp for numerical stability
            eta = np.clip(eta, -8.0, 8.0)
            mu = norm.cdf(eta)
            dmu = norm.pdf(eta)
            # Guard against extremes
            mu = np.clip(mu, 1e-6, 1 - 1e-6)
            dmu = np.clip(dmu, 1e-6, None)
            # working response and weights (standard probit GLM IRLS)
            z_work = eta + (y - mu) / dmu
            w = (dmu ** 2) / (mu * (1 - mu))
            # Clip weights to avoid numerical blow-up at extreme probabilities
            w = np.clip(w, 1e-6, 1e2)
            w = np.clip(w, 1e-6, None)

            # Jacobians wrt parameters
            Jb = self.utility.gradient_wrt_beta(X)  # (n,d)
            Ji = self.utility.gradient_wrt_intercept(X)  # (n,)
            J = np.hstack([Jb, Ji[:, None]])  # (n, d+1)

            # Solve for delta using normal equations with L2
            WJ = J * w[:, None]
            H = J.T @ WJ + self.l2 * np.eye(d + 1)
            rhs = J.T @ (w * (z_work - eta))
            try:
                delta = np.linalg.solve(H, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ rhs

            # Damped step to ensure stability
            step_factor = 0.5
            beta_new = beta + step_factor * delta[:d]
            intercept_new = intercept + step_factor * delta[d]
            step = np.max(np.abs(delta))
            beta, intercept = beta_new, intercept_new
            if step < self.tol:
                break

        self.utility.set_params(beta=beta, intercept=intercept)
        # compute nll
        eta = self.utility.evaluate(X)
        p = norm.cdf(eta)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        nll = -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return {"n_iter": it + 1, "nll": nll}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        eta = self.utility.evaluate(X)
        return norm.cdf(eta)


