import os
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from econ_model.utils import parse_code, is_yes, zscore_codes, MISSING_CODES


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def make_binary_work(df: pd.DataFrame) -> pd.Series:
    # ECON1: 1/2 表示工作（雇员/自雇），3 表示未工作，98/99 缺失
    codes = df["ECON1"].map(parse_code)
    y = np.where(codes.isin([1.0, 2.0]), 1.0, np.where(codes == 3.0, 0.0, np.nan))
    return pd.Series(y, index=df.index)


def build_features(df: pd.DataFrame, feature_set: str = "full") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # 特征集合：full（之前的较全特征）或 minimal（精简解释集）
    if feature_set == "minimal":
        # 精简可解释特征：10项
        cols_ord = ["AGE7", "PHYS8", "ECON4A", "ECON4B"]
        cols_cat = ["GENDER", "RACETH", "P_DENSE"]
        use_psych_score = True
        use_counts = True
        use_econ3 = True
        use_soc1_soc4b = False
        use_soc5_raw = False
    else:
        # 较全基线
        cols_ord = ["AGE7", "PHYS8", "ECON4A", "ECON4B", "SOC5A", "SOC5B", "SOC5C", "SOC5D", "SOC5E"]
        cols_cat = ["GENDER", "RACETH", "P_DENSE", "MODE", "LANGUAGE", "SOC1", "SOC4B"]
        use_psych_score = True
        use_counts = True
        use_econ3 = True
        use_soc1_soc4b = True
        use_soc5_raw = True
    # 注意：不纳入 REGION/P_GEO；高缺失的 P_OCCUPY2、MARITAL、LGBT 先不纳入基线

    X_parts: List[pd.DataFrame] = []

    # 有序：用代码值并标准化到 [z-score]（保序）
    for c in cols_ord:
        if c in df.columns:
            X_parts.append(zscore_codes(df[c]).rename(c))

    # 类别：One-Hot，保留缺失一类
    for c in cols_cat:
        if c in df.columns:
            code = df[c].map(parse_code)
            cat = code.where(~code.isin(MISSING_CODES), other=np.nan)
            d = pd.get_dummies(cat, prefix=c, dummy_na=True)
            X_parts.append(d)

    # 连续：ECON3（疫情前工时），做 winsorize 简单稳健化
    if use_econ3 and "ECON3" in df.columns:
        econ3 = df["ECON3"].map(parse_code)
        econ3 = econ3.where(~econ3.isin(MISSING_CODES))
        q1, q99 = econ3.quantile(0.01), econ3.quantile(0.99)
        econ3 = econ3.clip(lower=q1, upper=q99)
        econ3 = (econ3 - econ3.mean()) / (econ3.std() + 1e-8)
        X_parts.append(econ3.rename("ECON3_z"))

    # 派生特征：慢病计数、症状计数、心理分数
    phys3_cols = [c for c in df.columns if c.startswith("PHYS3") and len(c) == 6]
    if use_counts and phys3_cols:
        yes_matrix = pd.DataFrame({c: is_yes(df[c]) for c in phys3_cols})
        X_parts.append(yes_matrix.sum(axis=1).rename("chronic_condition_count"))

    phys1_cols = [c for c in df.columns if (c.startswith("PHYS1") and len(c) == 6) or c in {"PHYS7_1", "PHYS7_2", "PHYS7_3"}]
    if use_counts and phys1_cols:
        yes_matrix = pd.DataFrame({c: is_yes(df[c]) for c in phys1_cols})
        X_parts.append(yes_matrix.sum(axis=1).rename("symptom_count"))

    soc5_cols = [c for c in ["SOC5A", "SOC5B", "SOC5C", "SOC5D", "SOC5E"] if c in df.columns]
    if use_psych_score and soc5_cols:
        z_df = pd.DataFrame({c: zscore_codes(df[c]) for c in soc5_cols})
        X_parts.append(z_df.sum(axis=1).rename("psych_score"))
    # 若需要保留原 SOC5*（仅在 full）
    if use_soc5_raw and soc5_cols:
        for c in soc5_cols:
            X_parts.append(zscore_codes(df[c]).rename(c))

    # 可选：SOC1/SOC4B（仅在 full）
    if use_soc1_soc4b:
        for c in ["SOC1", "SOC4B"]:
            if c in df.columns:
                code = df[c].map(parse_code)
                cat = code.where(~code.isin(MISSING_CODES), other=np.nan)
                X_parts.append(pd.get_dummies(cat, prefix=c, dummy_na=True))

    X = pd.concat(X_parts, axis=1)
    # 用 0 填补缺失（有序与连续变量已标准化，0 约等于均值；分类提供了缺失指示列）
    X = X.fillna(0.0)

    # 目标
    y_work = make_binary_work(df)
    hours = df["ECON2"].map(parse_code)
    hours = hours.where(~hours.isin(MISSING_CODES))

    return X, y_work, hours


def save_numpy_dataset(X: pd.DataFrame, y_work: pd.Series, hours: pd.Series, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_cols.npy"), np.array(X.columns, dtype=object))
    np.save(os.path.join(out_dir, "X.npy"), X.to_numpy(dtype=float))
    np.save(os.path.join(out_dir, "y_work.npy"), y_work.to_numpy(dtype=float))
    np.save(os.path.join(out_dir, "hours.npy"), hours.to_numpy(dtype=float))
    return {"n_samples": int(len(X)), "n_features": int(X.shape[1])}


if __name__ == "__main__":
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "survey",
        "survey_results",
        "week1",
        "Covid_W1_NY.csv",
    )
    X, y_work, hours = build_features(load_csv(csv_path))
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    meta = save_numpy_dataset(X, y_work, hours, out_dir)
    print("Saved dataset:", meta)


