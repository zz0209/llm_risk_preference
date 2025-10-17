import os
import re
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_perception(weeks: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    root = os.path.join(PROJECT_ROOT, "outputs", "perception")
    pat = re.compile(r"^([A-Z]{2})_W(\d)_([0-9]+)\.txt$")
    for w in weeks:
        wk_dir = os.path.join(root, f"week{w}")
        if not os.path.isdir(wk_dir):
            continue
        for fn in os.listdir(wk_dir):
            if not fn.endswith(".txt"):
                continue
            m = pat.match(fn)
            if not m:
                continue
            state = m.group(1)
            week_in = int(m.group(2))
            su_id = m.group(3)
            if week_in != w:
                continue
            content = read_text(os.path.join(wk_dir, fn))
            try:
                p = float(content)
            except Exception:
                continue
            if not (0.0 <= p <= 1.0):
                continue
            rows.append({"STATE": state, "WEEK": w, "SU_ID": su_id, "perception": p})
    return pd.DataFrame(rows)


def load_llm_econ(weeks: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    root = os.path.join(PROJECT_ROOT, "outputs", "ECON")
    pat = re.compile(r"^([A-Z]{2})_W(\d)_([0-9]+)_([A-Z0-9_]+)\.txt$")
    for w in weeks:
        wk_dir = os.path.join(root, f"week{w}")
        if not os.path.isdir(wk_dir):
            continue
        for fn in os.listdir(wk_dir):
            if not fn.endswith(".txt"):
                continue
            m = pat.match(fn)
            if not m:
                continue
            state = m.group(1)
            week_in = int(m.group(2))
            su_id = m.group(3)
            var = m.group(4)
            if week_in != w or var not in {"ECON1", "ECON2"}:
                continue
            content = read_text(os.path.join(wk_dir, fn))
            rows.append({"STATE": state, "WEEK": w, "SU_ID": su_id, "Variable": var, "value": content})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["STATE", "WEEK", "SU_ID"], columns="Variable", values="value", aggfunc="first").reset_index()
    # parse ECON1 worked indicator (1 or 2 => worked)
    def econ1_worked(s: str) -> float:
        if not isinstance(s, str):
            return np.nan
        m = re.search(r"\((\d+)\)", s)
        if not m:
            return np.nan
        code = m.group(1)
        return 1.0 if code in {"1", "2"} else 0.0 if code in {"3"} else np.nan
    wide["ECON1_worked"] = wide.get("ECON1").apply(econ1_worked) if "ECON1" in wide.columns else np.nan
    # parse ECON2 hours
    def econ2_hours(s: str) -> float:
        if not isinstance(s, str):
            return np.nan
        s = s.strip()
        if re.match(r"^\d+$", s):
            v = int(s)
            return float(v) if 0 <= v <= 168 else np.nan
        m = re.match(r"\s*(\d+)", s)
        if m:
            v = int(m.group(1))
            return float(v) if 0 <= v <= 168 else np.nan
        return np.nan
    wide["ECON2_hours"] = wide.get("ECON2").apply(econ2_hours) if "ECON2" in wide.columns else np.nan
    return wide


def per_id_metrics(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    # group by (STATE, SU_ID)
    rows: List[Dict[str, float]] = []
    for (state, su), g in df.groupby(["STATE", "SU_ID" ]):
        g = g.sort_values("WEEK")
        # ECON1: correlation / OLS between perception and worked indicator
        r1 = np.nan; slope1 = np.nan; r2_1 = np.nan; n1 = 0
        if g["ECON1_worked"].notna().sum() >= 2:
            x = g.loc[g["ECON1_worked"].notna(), "perception"].astype(float).values
            y = g.loc[g["ECON1_worked"].notna(), "ECON1_worked"].astype(float).values
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
                r1 = float(np.corrcoef(x, y)[0,1])
            # OLS y = a + b x
            if len(x) >= 2:
                X = np.vstack([np.ones_like(x), x]).T
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    yhat = X @ beta
                    ss_res = float(np.sum((y - yhat) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    r2_1 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                    slope1 = float(beta[1])
                except Exception:
                    pass
            n1 = int(len(x))
        # ECON2: hours conditional correlation / OLS
        r2 = np.nan; slope2 = np.nan; r2_2 = np.nan; n2 = 0
        sub = g[g["ECON2_hours"].notna()]
        if len(sub) >= 2:
            x = sub["perception"].astype(float).values
            y = sub["ECON2_hours"].astype(float).values
            if np.std(x) > 0 and np.std(y) > 0:
                r2 = float(np.corrcoef(x, y)[0,1])
            X = np.vstack([np.ones_like(x), x]).T
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                yhat = X @ beta
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2_2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                slope2 = float(beta[1])
            except Exception:
                pass
            n2 = int(len(x))
        rows.append({
            "STATE": state,
            "SU_ID": su,
            "n_obs": int(len(g)),
            "econ1_n": n1,
            "econ1_corr": r1,
            "econ1_slope": slope1,
            "econ1_r2": r2_1,
            "econ2_n": n2,
            "econ2_corr": r2,
            "econ2_slope": slope2,
            "econ2_r2": r2_2,
        })
    out_csv = os.path.join(out_dir, "per_id_explainability.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Visualizations of per-id correlations
    dfm = pd.DataFrame(rows)
    # ECON1 corr
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(dfm["econ1_corr"].dropna().values, bins=30, color="#4C78A8", edgecolor="white")
    ax.set_title("Per-ID correlation: perception vs ECON1(worked)")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_corr_econ1.png"), dpi=200)
    plt.close(fig)
    # ECON2 corr
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(dfm["econ2_corr"].dropna().values, bins=30, color="#F58518", edgecolor="white")
    ax.set_title("Per-ID correlation: perception vs ECON2(hours)")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_corr_econ2.png"), dpi=200)
    plt.close(fig)

    # Boxplots by state
    for col, name in [("econ1_corr", "econ1_corr"), ("econ2_corr", "econ2_corr")]:
        fig, ax = plt.subplots(figsize=(6,4))
        data = [dfm.loc[dfm["STATE"]==st, col].dropna().values for st in ["NY", "TX"]]
        ax.boxplot(data, labels=["NY","TX"], vert=True, showfliers=False)
        ax.set_title(f"Per-ID {col} by state")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"box_{name}_by_state.png"), dpi=200)
        plt.close(fig)


def cross_sectional_metrics(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows_all: List[Dict[str, float]] = []
    rows_state: List[Dict[str, float]] = []

    def fit_metrics(sub: pd.DataFrame) -> Tuple[float, float, int, float, float, int]:
        # ECON1 binary vs perception
        econ1_corr = np.nan; econ1_r2 = np.nan; n1 = 0
        s1 = sub.dropna(subset=["perception", "ECON1_worked"]) if "ECON1_worked" in sub.columns else sub.iloc[0:0]
        if not s1.empty:
            x = s1["perception"].astype(float).values
            y = s1["ECON1_worked"].astype(float).values
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
                econ1_corr = float(np.corrcoef(x, y)[0,1])
            if len(x) >= 2:
                X = np.vstack([np.ones_like(x), x]).T
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    yhat = X @ beta
                    ss_res = float(np.sum((y - yhat) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    econ1_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                except Exception:
                    pass
            n1 = int(len(x))
        # ECON2 hours vs perception（仅 worked 有小时的）
        econ2_corr = np.nan; econ2_r2 = np.nan; n2 = 0
        s2 = sub.dropna(subset=["perception", "ECON2_hours"]) if "ECON2_hours" in sub.columns else sub.iloc[0:0]
        if not s2.empty:
            x = s2["perception"].astype(float).values
            y = s2["ECON2_hours"].astype(float).values
            if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
                econ2_corr = float(np.corrcoef(x, y)[0,1])
            if len(x) >= 2:
                X = np.vstack([np.ones_like(x), x]).T
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    yhat = X @ beta
                    ss_res = float(np.sum((y - yhat) ** 2))
                    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                    econ2_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                except Exception:
                    pass
            n2 = int(len(x))
        return econ1_corr, econ1_r2, n1, econ2_corr, econ2_r2, n2

    # pooled across all weeks
    m_all = fit_metrics(df)
    rows_all.append({
        "scope": "pooled",
        "econ1_corr": m_all[0], "econ1_r2": m_all[1], "econ1_n": m_all[2],
        "econ2_corr": m_all[3], "econ2_r2": m_all[4], "econ2_n": m_all[5],
    })

    # by state (pooled weeks)
    for st, g in df.groupby("STATE"):
        m = fit_metrics(g)
        rows_state.append({
            "STATE": st,
            "econ1_corr": m[0], "econ1_r2": m[1], "econ1_n": m[2],
            "econ2_corr": m[3], "econ2_r2": m[4], "econ2_n": m[5],
        })

    pd.DataFrame(rows_all).to_csv(os.path.join(out_dir, "pooled_metrics_overall.csv"), index=False)
    pd.DataFrame(rows_state).to_csv(os.path.join(out_dir, "pooled_metrics_by_state.csv"), index=False)

    # Plots
    # ECON2 pooled scatter with OLS
    sub = df.dropna(subset=["perception", "ECON2_hours"]) if "ECON2_hours" in df.columns else df.iloc[0:0]
    if len(sub) >= 5:
        x = sub["perception"].astype(float).values
        y = sub["ECON2_hours"].astype(float).values
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(x, y, s=10, alpha=0.25, color="#4C72B0")
        try:
            X = np.vstack([np.ones_like(x), x]).T
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            xs = np.linspace(min(x), max(x), 200)
            ys = beta[0] + beta[1] * xs
            ax.plot(xs, ys, color="#333333", linestyle="--")
        except Exception:
            pass
        ax.set_xlabel("Perception (0-1)")
        ax.set_ylabel("ECON2 hours")
        ax.set_title("ECON2 vs Perception (pooled)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "scatter_econ2_vs_perception_pooled.png"), dpi=200)
        plt.close(fig)

    # Bar correlations by state
    dfs = pd.DataFrame(rows_state)
    if not dfs.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(dfs["STATE"], dfs["econ1_corr"], color="#4C78A8")
        ax.set_title("Perception -> ECON1 corr (pooled by state)")
        ax.set_ylabel("Pearson r")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bar_corr_econ1_by_state.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(dfs["STATE"], dfs["econ2_corr"], color="#F58518")
        ax.set_title("Perception -> ECON2 corr (pooled by state)")
        ax.set_ylabel("Pearson r")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bar_corr_econ2_by_state.png"), dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=str, default="1,2,3")
    args = parser.parse_args()
    weeks = [int(w.strip()) for w in args.weeks.split(",") if w.strip()]

    per_dir = os.path.join(PROJECT_ROOT, "outputs", "perception")
    econ_dir = os.path.join(PROJECT_ROOT, "outputs", "ECON")
    assert os.path.isdir(per_dir), "perception outputs not found"
    assert os.path.isdir(econ_dir), "ECON outputs not found"

    per_df = load_perception(weeks)
    econ_df = load_llm_econ(weeks)
    if per_df.empty or econ_df.empty:
        print("No perception/econ data found.")
        return
    # merge
    df = per_df.merge(econ_df, on=["STATE","WEEK","SU_ID"], how="left")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "diagrams", "perception_explainability")
    ensure_dir(out_dir)
    # Save merged
    df.to_csv(os.path.join(out_dir, "merged_perception_econ.csv"), index=False)
    per_id_metrics(df, out_dir)
    pooled_out = os.path.join(out_dir, "pooled")
    cross_sectional_metrics(df, pooled_out)
    print(json.dumps({"merged_rows": int(len(df))}))


if __name__ == "__main__":
    main()


