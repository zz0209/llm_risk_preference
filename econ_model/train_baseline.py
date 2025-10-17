import os
from typing import Dict, Any

import numpy as np

from econ_model.data_prep import load_csv, build_features
from econ_model.utility import LinearUtility, ProspectUtility
from econ_model.decision import BinaryLogit, OLS
from econ_model.utils import to_json


def split_mask(n: int, seed: int = 42, test_ratio: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_ratio))
    mask = np.zeros(n, dtype=bool)
    mask[idx[:cut]] = True
    return mask


def evaluate_classification(y_true: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    y_true = y_true.astype(float)
    keep = ~np.isnan(y_true)
    y = y_true[keep]
    p = p[keep]
    yhat = (p >= 0.5).astype(float)
    acc = float((yhat == y).mean())
    brier = float(np.mean((p - y) ** 2))
    # 近似 logloss
    logloss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return {"acc": acc, "brier": brier, "logloss": logloss}


def load_weeks(root: str) -> np.ndarray:
    # 读取 Full CSV：W1/W2/W3
    paths = [
        os.path.join(root, "survey", "survey_results", "week1", "Covid_W1_Full.csv"),
        os.path.join(root, "survey", "survey_results", "week2", "Covid_W2_Full.csv"),
        os.path.join(root, "survey", "survey_results", "week3", "Covid_W3_Full.csv"),
    ]
    return np.array(paths)


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 构建特征与标签（合并 W1-3 Full）并使用精简特征集
    week_paths = load_weeks(root)
    import pandas as pd
    dfs = [load_csv(p) for p in week_paths if os.path.exists(p)]
    if not dfs:
        # 回退到 W1_NY 样本
        csv_path = os.path.join(root, "survey", "survey_results", "week1", "Covid_W1_NY.csv")
        dfs = [load_csv(csv_path)]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    X_df, y_work_s, hours_s = build_features(df, feature_set="minimal")
    X = X_df.to_numpy(dtype=float)
    y_work = y_work_s.to_numpy(dtype=float)
    hours = hours_s.to_numpy(dtype=float)

    # 训练/验证划分（仅对有标签样本）
    keep_cls = ~np.isnan(y_work)
    Xc, yc = X[keep_cls], y_work[keep_cls]
    mask = split_mask(len(Xc))
    Xtr, Xte = Xc[mask], Xc[~mask]
    ytr, yte = yc[mask], yc[~mask]

    # 二元 Logit（线性效用）
    # 线性效用模型
    util_lin = LinearUtility(n_features=X.shape[1])
    clf_lin = BinaryLogit(util_lin, l2=1e-2, max_iter=200)
    train_info_lin = clf_lin.fit(Xtr, ytr)
    p_tr_lin = clf_lin.predict_proba(Xtr)
    p_te_lin = clf_lin.predict_proba(Xte)
    cls_tr_lin = evaluate_classification(ytr, p_tr_lin)
    cls_te_lin = evaluate_classification(yte, p_te_lin)
    util_params_lin = util_lin.get_params()
    beta_lin = np.array(util_params_lin["beta"], dtype=float)
    intercept_lin = float(util_params_lin["intercept"])
    coef_map_lin = {str(col): float(w) for col, w in zip(X_df.columns.tolist(), beta_lin)}

    # Prospect Theory 效用模型（参数用文献常用起点，可进一步调优）
    util_pt = ProspectUtility(n_features=X.shape[1])
    clf_pt = BinaryLogit(util_pt, l2=1e-2, max_iter=200)
    train_info_pt = clf_pt.fit(Xtr, ytr)
    p_tr_pt = clf_pt.predict_proba(Xtr)
    p_te_pt = clf_pt.predict_proba(Xte)
    cls_tr_pt = evaluate_classification(ytr, p_tr_pt)
    cls_te_pt = evaluate_classification(yte, p_te_pt)
    util_params_pt = util_pt.get_params()
    beta_pt = np.array(util_params_pt["beta"], dtype=float)
    intercept_pt = float(util_params_pt["intercept"])
    coef_map_pt = {str(col): float(w) for col, w in zip(X_df.columns.tolist(), beta_pt)}

    # 小时模型：仅对 y=1 子样本
    keep_hours = (y_work == 1.0) & ~np.isnan(hours)
    Xh, yh = X[keep_hours], hours[keep_hours]
    mask_h = split_mask(len(Xh))
    Xh_tr, Xh_te = Xh[mask_h], Xh[~mask_h]
    yh_tr, yh_te = yh[mask_h], yh[~mask_h]
    reg = OLS()
    reg_info = reg.fit(Xh_tr, yh_tr)
    yh_pred = reg.predict(Xh_te)
    rmse = float(np.sqrt(np.mean((yh_pred - yh_te) ** 2)))
    mae = float(np.mean(np.abs(yh_pred - yh_te)))
    # 提取回归系数
    hours_coef_map = {str(col): float(w) for col, w in zip(X_df.columns.tolist(), reg.coef_)}

    # 保存结果
    out = {
        "classification_linear": {
            "train": {**train_info_lin, **cls_tr_lin},
            "test": cls_te_lin,
            "coef_norm": train_info_lin.get("beta_norm"),
            "intercept": intercept_lin,
            "coef": coef_map_lin,
        },
        "classification_prospect": {
            "train": {**train_info_pt, **cls_tr_pt},
            "test": cls_te_pt,
            "coef_norm": train_info_pt.get("beta_norm"),
            "intercept": intercept_pt,
            "coef": coef_map_pt,
            "pt_params": {k: util_params_pt[k] for k in ["alpha", "beta_v", "lambda", "pw_gamma"]},
        },
        "hours": {
            "train": reg_info,
            "test": {"rmse": rmse, "mae": mae},
            "intercept": float(reg.intercept_),
            "coef": hours_coef_map,
        },
        "shapes": {
            "X": [int(X.shape[0]), int(X.shape[1])],
            "X_hours": [int(Xh.shape[0]), int(Xh.shape[1])],
        },
    }

    to_json(out, os.path.join(results_dir, "baseline_summary.json"))
    # 也保存真实与预测（用于可视化）
    np.save(os.path.join(results_dir, "work_proba_test.npy"), p_te_lin)
    np.save(os.path.join(results_dir, "work_proba_test_pt.npy"), p_te_pt)
    np.save(os.path.join(results_dir, "work_y_test.npy"), yte)
    np.save(os.path.join(results_dir, "hours_pred_test.npy"), yh_pred)
    np.save(os.path.join(results_dir, "hours_y_test.npy"), yh_te)

    print("Saved results to:", results_dir)


if __name__ == "__main__":
    main()


