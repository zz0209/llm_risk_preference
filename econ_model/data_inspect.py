import json
import os
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np


CODEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "survey",
    "survey_results",
    "final_questionnaire.json",
)


def load_codebook(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_variable_index(codebook: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for item in codebook:
        var = item.get("Variable")
        if not var:
            continue
        idx[var] = {
            "label": item.get("Variable Label"),
            "category": item.get("Category"),
            "options": item.get("options", []),
            "condition": item.get("Condition"),
        }
    return idx


def summarize_series(s: pd.Series) -> Dict[str, Any]:
    total = len(s)
    missing_vals = {77, 98, 99, 777, 888, 998, 999}
    n_missing = int(s.isna().sum()) + int(s.isin(missing_vals).sum() if s.dtype != object else 0)
    unique_vals = s.dropna().unique()
    nunique = int(s.nunique(dropna=True))
    return {
        "total": total,
        "nunique": nunique,
        "n_missing_marked": n_missing,
        "missing_rate": round(n_missing / total, 4) if total > 0 else None,
        "sample_values": ", ".join(map(str, unique_vals[:10]))
        if len(unique_vals) > 0
        else "",
    }


def choose_preferred_age(columns: List[str]) -> Tuple[str, str]:
    # Prefer AGE7 over AGE4 for finer granularity; fall back if not present
    if "AGE7" in columns:
        return "AGE7", "优先使用 AGE7（7 档），若缺失或样本过稀再回退 AGE4。"
    if "AGE4" in columns:
        return "AGE4", "AGE7 不存在，使用 AGE4（4 档）。"
    return "", "未找到 AGE 变量。"


def category_encoding_suggestion(var: str, meta: Dict[str, Any]) -> str:
    category = meta.get("category")
    options = meta.get("options", [])
    if not options:
        return "连续/自由文本或缺少选项；视分布而定（标准化或分箱）。"
    # Ordinal candidates
    ordinal_candidates = {"AGE4", "AGE7", "PHYS8", "SOC5A", "SOC5B", "SOC5C", "SOC5D", "SOC5E", "ECON4A", "ECON4B"}
    if var in ordinal_candidates:
        return "有序编码（保留等级顺序），并对缺失码另设指示器。"
    # High-cardinality geography/occupation
    if var in {"P_GEO", "P_OCCUPY2"}:
        return "类别较多：建议分层/随机效应或目标编码；或用 REGION4 替代 P_GEO 基线。"
    # Binary/multi-class small
    values = {opt.get("Value") for opt in options}
    if values.issubset({0, 1}) or len(values) <= 6:
        return "One-Hot 编码；77/98/99 视为缺失并加缺失指示器。"
    return "One-Hot 或目标编码（视基数与样本量）；缺失码单独处理。"


MISSING_CODES = {77, 98, 99, 777, 888, 998, 999}


def parse_code(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # Expect format like "(1) Yes" or "1" or "(0) No"
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


def zscore(series: pd.Series) -> pd.Series:
    # Parse coded strings like "(1) ..." to numeric codes, set special missing codes to NaN
    x = series.map(parse_code)
    x = x.where(~x.isin(MISSING_CODES))
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.nan, index=series.index)
    return (x - mu) / sd


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inspect persona variables and encoding suggestions.")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "survey",
            "survey_results",
            "week1",
            "Covid_W1_NY.csv",
        ),
        help="Path to a sample CSV file to inspect",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "reports",
            "variables_summary.csv",
        ),
        help="Path to write variables summary CSV",
    )
    parser.add_argument(
        "--derived_out",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "reports",
            "derived_summary.csv",
        ),
        help="Path to write derived features distribution summary",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    codebook = load_codebook(CODEBOOK_PATH)
    var_index = build_variable_index(codebook)

    # Load CSV (assume comma-separated, UTF-8)
    df = pd.read_csv(args.csv)

    # Candidate variable groups
    core_vars: List[str] = [
        # Demographics & household
        "AGE7", "AGE4", "GENDER", "RACETH", "HHINCOME", "EDUC4", "EDUCATION",
        "HHSIZE1", "HH01S", "HH25S", "HH612S", "HH1317S", "HH18OVS", "MARITAL", "LGBT",
        # Geography & mode
        "REGION4", "REGION9", "P_GEO", "P_DENSE", "MODE", "LANGUAGE",
        # Occupation
        "P_OCCUPY2",
        # Health & symptoms
        "PHYS8",
        "PHYS3A", "PHYS3B", "PHYS3C", "PHYS3D", "PHYS3E", "PHYS3F", "PHYS3G", "PHYS3H", "PHYS3I", "PHYS3J", "PHYS3K", "PHYS3L", "PHYS3M",
        "PHYS1A", "PHYS1B", "PHYS1C", "PHYS1D", "PHYS1E", "PHYS1F", "PHYS1G", "PHYS1H", "PHYS1I", "PHYS1J", "PHYS1K", "PHYS1L", "PHYS1M", "PHYS1N", "PHYS1O", "PHYS1P", "PHYS1Q",
        "PHYS7_1", "PHYS7_2", "PHYS7_3", "PHYS7_4", "PHYS7_DK", "PHYS7_SKP", "PHYS7_REF",
        "PHYS4", "PHYS5", "PHYS6",
        # Psychological
        "SOC5A", "SOC5B", "SOC5C", "SOC5D", "SOC5E",
        # Social capital
        "SOC1", "SOC4B",
        # Employment baseline/expectations
        "ECON3", "ECON4A", "ECON4B",
        # Targets (for context only; not encoded here)
        "ECON1", "ECON2",
    ]

    present_columns = [c for c in core_vars if c in df.columns]
    missing_columns = [c for c in core_vars if c not in df.columns]

    preferred_age, age_note = choose_preferred_age(df.columns.tolist())

    rows: List[Dict[str, Any]] = []

    for var in present_columns:
        meta = var_index.get(var, {"label": None, "category": None, "options": []})
        s = df[var]
        summary = summarize_series(s)
        suggestion = category_encoding_suggestion(var, meta)
        rows.append(
            {
                "variable": var,
                "label": meta.get("label"),
                "category": meta.get("category"),
                **summary,
                "encoding_suggestion": suggestion,
            }
        )

    # Add notes for key duplicates/choices
    rows.append(
        {
            "variable": "AGE_choice",
            "label": "AGE4 vs AGE7",
            "category": "persona",
            "total": None,
            "nunique": None,
            "n_missing_marked": None,
            "missing_rate": None,
            "sample_values": age_note,
            "encoding_suggestion": f"使用 {preferred_age} 作为主变量；另一项可作一致性检验。",
        }
    )

    # Proposed derived features presence checks
    derived_notes = [
        ("chronic_condition_count", "由 PHYS3A–PHYS3M 计数（Yes=1）"),
        ("symptom_count", "由 PHYS1A–PHYS1Q 与 PHYS7_* 计数（Yes=1）"),
        ("psych_score", "SOC5A–E 标准化后求和或 PCA 第一主成分"),
    ]
    for name, note in derived_notes:
        rows.append(
            {
                "variable": name,
                "label": note,
                "category": "derived_proposal",
                "total": None,
                "nunique": None,
                "n_missing_marked": None,
                "missing_rate": None,
                "sample_values": "待计算（此脚本不生成派生列）",
                "encoding_suggestion": "后续在数据准备阶段构造并标准化。",
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.out, index=False)

    # Derived features distribution (no column creation, just summary on-the-fly)
    derived_rows: List[Dict[str, Any]] = []

    # chronic_condition_count: PHYS3A–PHYS3M Yes=1 count
    phys3_cols = [c for c in df.columns if c.startswith("PHYS3") and len(c) == 6]
    if phys3_cols:
        yes_matrix = pd.DataFrame({c: is_yes(df[c]) for c in phys3_cols})
        chronic_count = yes_matrix.sum(axis=1)
        derived_rows.append({
            "feature": "chronic_condition_count",
            "min": float(chronic_count.min()),
            "max": float(chronic_count.max()),
            "mean": float(chronic_count.mean()),
            "std": float(chronic_count.std()),
            "p50": float(chronic_count.quantile(0.5)),
            "p90": float(chronic_count.quantile(0.9)),
        })

    # symptom_count: PHYS1A–PHYS1Q and PHYS7_* Yes=1 count
    phys1_cols = [
        c for c in df.columns
        if (c.startswith("PHYS1") and len(c) == 6) or c in {"PHYS7_1", "PHYS7_2", "PHYS7_3"}
    ]
    if phys1_cols:
        yes_matrix = pd.DataFrame({c: is_yes(df[c]) for c in phys1_cols})
        symptom_count = yes_matrix.sum(axis=1)
        derived_rows.append({
            "feature": "symptom_count",
            "min": float(symptom_count.min()),
            "max": float(symptom_count.max()),
            "mean": float(symptom_count.mean()),
            "std": float(symptom_count.std()),
            "p50": float(symptom_count.quantile(0.5)),
            "p90": float(symptom_count.quantile(0.9)),
        })

    # psych_score: zscore(SOC5A–E) sum
    soc5_cols = [c for c in ["SOC5A", "SOC5B", "SOC5C", "SOC5D", "SOC5E"] if c in df.columns]
    if soc5_cols:
        z_df = pd.DataFrame({c: zscore(df[c]) for c in soc5_cols})
        psych_score = z_df.sum(axis=1, skipna=True)
        derived_rows.append({
            "feature": "psych_score(zsum)",
            "min": float(psych_score.min()),
            "max": float(psych_score.max()),
            "mean": float(psych_score.mean()),
            "std": float(psych_score.std()),
            "p50": float(psych_score.quantile(0.5)),
            "p90": float(psych_score.quantile(0.9)),
        })

    if derived_rows:
        pd.DataFrame(derived_rows).to_csv(args.derived_out, index=False)

    # Also print brief console summary
    print("Loaded codebook with variables:", len(var_index))
    print("CSV path:", args.csv)
    print("Columns present:", len(df.columns))
    print("Persona candidates present:", len(present_columns))
    print("Persona candidates missing:", len(missing_columns))
    if missing_columns:
        print("Missing (first 20):", ", ".join(missing_columns[:20]))
    print("Preferred age variable:", preferred_age, "|", age_note)
    print("Summary written to:", args.out)
    if derived_rows:
        print("Derived summary written to:", args.derived_out)


if __name__ == "__main__":
    main()


