import json
from typing import Any, Set

import numpy as np
import pandas as pd


MISSING_CODES: Set[int] = {77, 98, 99, 777, 888, 998, 999}


def parse_code(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s.startswith("(") and ")" in s:
        try:
            num = s[1 : s.index(")")]
            return float(num)
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def is_yes(series: pd.Series, yes_code: int = 1) -> pd.Series:
    codes = series.map(parse_code)
    return (codes == yes_code).astype(float)


def zscore_codes(series: pd.Series) -> pd.Series:
    codes = series.map(parse_code)
    codes = codes.where(~codes.isin(MISSING_CODES))
    mu = codes.mean(skipna=True)
    sd = codes.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.nan, index=series.index)
    return (codes - mu) / sd


def to_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


