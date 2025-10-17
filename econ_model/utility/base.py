from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class UtilityModel(ABC):
    """抽象效用接口，允许替换为 EUT、PT 等不同规范化。

    要求实现 evaluate(X) -> utility 的前向计算；
    参数拟合则交给 decision 层（如 Logit）通过梯度/优化完成。
    """

    @abstractmethod
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def set_params(self, **kwargs: Any) -> None:
        raise NotImplementedError

    # 可选：提供对参数的梯度映射，以支持非线性效用的链式法则
    def gradient_wrt_beta(self, X: np.ndarray) -> np.ndarray:
        # 默认线性：grad = X
        return X

    def gradient_wrt_intercept(self, X: np.ndarray) -> np.ndarray:
        # 默认线性：grad = 1
        return np.ones(X.shape[0])


