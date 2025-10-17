from typing import Dict, Any, Optional

import numpy as np

from econ_model.utility import UtilityModel


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class BinaryLogit:
    """二元 Logit：P(y=1|X) = sigmoid(U(X)).

    - 通过极大似然拟合 Utility 的参数（线性效用时等价于逻辑回归）。
    - 使用牛顿-拉弗森（或拟牛顿）更新；带 L2 正则。
    - 可替换任意 UtilityModel，只要其 evaluate 可由线性形式参数化。
    """

    def __init__(self, utility: UtilityModel, l2: float = 1e-3, max_iter: int = 100, tol: float = 1e-6):
        self.utility = utility
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_features: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n, d = X.shape
        self.n_features = d

        # 初始化参数
        params = self.utility.get_params()
        beta = np.zeros(d) if params.get("beta") is None else np.asarray(params["beta"], dtype=float)
        intercept = float(params.get("intercept", 0.0))

        for it in range(self.max_iter):
            # 通过 utility 的前向计算得到系统效用
            self.utility.set_params(beta=beta, intercept=intercept)
            z = self.utility.evaluate(X)
            p = sigmoid(z)
            # 负对数似然 + L2
            # grad_beta = X^T (p - y) + l2 * beta
            r = p - y
            # 非线性效用的链式法则： grad_beta = (dU/dbeta)^T (p - y)
            J_beta = self.utility.gradient_wrt_beta(X)  # shape (n,d)
            grad_beta = J_beta.T @ r + self.l2 * beta
            grad_intercept = float((self.utility.gradient_wrt_intercept(X) * r).sum())

            # Hessian: X^T W X + l2 I, where W=diag(p*(1-p))
            w = p * (1 - p)
            # 为数值稳定，加入微小下界
            w = np.clip(w, 1e-6, None)
            # 近似 Hessian：J^T W J + l2 I
            WX = J_beta * w[:, None]
            H = J_beta.T @ WX + self.l2 * np.eye(d)

            # 联合更新：把截距并入增广向量
            H_aug = np.zeros((d + 1, d + 1))
            H_aug[:d, :d] = H
            H_aug[d, d] = float((self.utility.gradient_wrt_intercept(X) ** 2 * w).sum())
            g_aug = np.concatenate([grad_beta, np.array([grad_intercept])])

            try:
                step = np.linalg.solve(H_aug, g_aug)
            except np.linalg.LinAlgError:
                # 回退到小步长梯度下降
                step = g_aug * 1e-3

            beta_new = beta - step[:d]
            intercept_new = intercept - step[d]

            delta = np.max(np.abs(step))
            beta, intercept = beta_new, intercept_new
            if delta < self.tol:
                break

        # 设置参数回 utility
        self.utility.set_params(beta=beta, intercept=intercept)

        # 最终指标
        z = X @ beta + intercept
        p = sigmoid(z)
        eps = 1e-12
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 0.5 * self.l2 * float(beta @ beta)
        return {
            "n_iter": it + 1,
            "nll": float(nll),
            "beta_norm": float(np.linalg.norm(beta)),
            "intercept": float(intercept),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        params = self.utility.get_params()
        beta = np.asarray(params["beta"], dtype=float)
        intercept = float(params["intercept"])
        z = X @ beta + intercept
        return sigmoid(z)


