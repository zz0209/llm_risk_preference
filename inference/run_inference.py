import os
import re
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_code_from_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    m = re.search(r"\((\d+)\)", s)
    return m.group(1) if m else ""


def load_human_week(week: int) -> pd.DataFrame:
    rows = []
    for state in ["NY", "TX"]:
        csv_path = os.path.join(PROJECT_ROOT, "survey", "survey_results", f"week{week}", f"Covid_W{week}_{state}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, dtype=str)
        df["STATE"] = state
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    # Ensure columns exist
    if "ECON1" not in df.columns:
        df["ECON1"] = np.nan
    if "ECON2" not in df.columns:
        df["ECON2"] = np.nan
    # ECON1 code is categorical; ECON2 should be numeric hours when applicable
    df["ECON1_code"] = df["ECON1"].apply(parse_code_from_text)
    # Pre-parse Human ECON2 hours
    def human_hours(x):
        if not isinstance(x, str):
            return np.nan
        x = x.strip()
        if x.isdigit():
            v = int(x)
            return float(v) if 0 <= v <= 168 else np.nan
        m = re.match(r"\s*(\d+)", x)
        if m:
            v = int(m.group(1))
            return float(v) if 0 <= v <= 168 else np.nan
        return np.nan
    df["ECON2_hours_Human"] = df["ECON2"].apply(human_hours) if "ECON2" in df.columns else np.nan
    return df


def load_llm_week(week: int) -> pd.DataFrame:
    econ_dir = os.path.join(PROJECT_ROOT, "outputs", "ECON", f"week{week}")
    if not os.path.isdir(econ_dir):
        return pd.DataFrame()
    records: List[Dict[str, str]] = []
    pat = re.compile(r"^([A-Z]{2})_W(\d)_([0-9]+)_([A-Z0-9_]+)\.txt$")
    for fn in os.listdir(econ_dir):
        if not fn.endswith(".txt"):
            continue
        m = pat.match(fn)
        if not m:
            continue
        state = m.group(1)
        week_in_name = int(m.group(2))
        su_id = m.group(3)
        var = m.group(4)
        if week_in_name != week or var not in {"ECON1", "ECON2"}:
            continue
        content = read_text(os.path.join(econ_dir, fn)).strip()
        rec = {"STATE": state, "SU_ID": su_id, "Variable": var, "LLM_Answer": content}
        records.append(rec)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # pivot to wide per SU_ID
    pivot = df.pivot_table(index=["STATE", "SU_ID"], columns="Variable", values="LLM_Answer", aggfunc="first").reset_index()
    # parse codes
    if "ECON1" in pivot.columns:
        pivot["ECON1_code_LLM"] = pivot["ECON1"].apply(parse_code_from_text)
    if "ECON2" in pivot.columns:
        def parse_hours(x: str):
            if not isinstance(x, str):
                return np.nan
            x = x.strip()
            if x.isdigit():
                v = int(x)
                if 0 <= v <= 168:
                    return v
            # try to extract leading integer
            m = re.match(r"\s*(\d+)", x)
            if m:
                v = int(m.group(1))
                if 0 <= v <= 168:
                    return v
            return np.nan
        pivot["ECON2_hours_LLM"] = pivot["ECON2"].apply(parse_hours) if "ECON2" in pivot.columns else np.nan
    return pivot


def merge_llm_human(week: int) -> pd.DataFrame:
    human = load_human_week(week)
    llm = load_llm_week(week)
    if human.empty or llm.empty:
        return pd.DataFrame()
    # Join full human columns to keep persona variables available
    hsel = human.copy()
    df = hsel.merge(llm, on=["STATE", "SU_ID"], how="left", suffixes=("_human", "_llm"))
    return df


def merge_llm_human_matched(week: int, matched_only: bool = False) -> pd.DataFrame:
    df = merge_llm_human(week)
    if df.empty:
        return df
    if matched_only:
        # 仅保留存在 LLM ECON1 的样本（表示 LLM 实际参与了该 SU_ID）
        if "ECON1_code_LLM" in df.columns:
            df = df[df["ECON1_code_LLM"].notna()]
        else:
            # 若不存在该列，则无任何匹配样本
            df = df.iloc[0:0]
    return df


def figure_folder(name: str) -> str:
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "diagrams", name)
    ensure_dir(out_dir)
    return out_dir


def figure_name(name: str, matched_only: bool) -> str:
    return f"{name}_matched" if matched_only else name


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def plot_hist_econ1(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON1_hist", matched_only)
    out = figure_folder(name)
    all_rows = []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        df["WEEK"] = w
        all_rows.append(df)
    if not all_rows:
        return
    df_all = pd.concat(all_rows, ignore_index=True)

    # Prepare counts
    econ1_map = {"1": "Yes employee", "2": "Self-employed", "3": "No", "": "Unknown"}
    df_all["ECON1_code_LLM_lbl"] = df_all["ECON1_code_LLM"].fillna("").astype(str).map(econ1_map).fillna("Other")
    df_all["ECON1_code_human_lbl"] = df_all["ECON1_code"].fillna("").astype(str).map(econ1_map).fillna("Other")

    for label, col in [("LLM", "ECON1_code_LLM_lbl"), ("Human", "ECON1_code_human_lbl")]:
        fig, ax = plt.subplots(figsize=(6,4))
        counts = df_all[col].value_counts().reindex(["Yes employee", "Self-employed", "No", "Unknown", "Other"], fill_value=0)
        counts.plot(kind='bar', ax=ax, color="#4C72B0")
        ax.set_title(f"ECON1 {label} histogram (W1+W2)")
        ax.set_ylabel("Count")
        ax.set_xlabel("Category")
        fig.tight_layout()
        fig_path = os.path.join(out, f"econ1_{label.lower()}_hist_overall.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        # Save data
        save_csv(counts.reset_index().rename(columns={"index":"category", col:"count"}), os.path.join(out, f"econ1_{label.lower()}_hist_overall.csv"))


def plot_hist_econ1_per_week(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON1_hist_per_week", matched_only)
    out = figure_folder(name)
    econ1_map = {"1": "Yes employee", "2": "Self-employed", "3": "No", "": "Unknown"}
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        df["ECON1_code_LLM_lbl"] = df.get("ECON1_code_LLM").fillna("").astype(str).map(econ1_map).fillna("Other")
        df["ECON1_code_human_lbl"] = df.get("ECON1_code").fillna("").astype(str).map(econ1_map).fillna("Other")
        for label, col in [("LLM", "ECON1_code_LLM_lbl"), ("Human", "ECON1_code_human_lbl")]:
            fig, ax = plt.subplots(figsize=(6,4))
            counts = df[col].value_counts().reindex(["Yes employee", "Self-employed", "No", "Unknown", "Other"], fill_value=0)
            counts.plot(kind='bar', ax=ax, color="#4C72B0")
            ax.set_title(f"ECON1 {label} histogram (W{w})")
            ax.set_ylabel("Count")
            ax.set_xlabel("Category")
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"econ1_{label.lower()}_hist_W{w}.png"), dpi=200)
            plt.close(fig)
            save_csv(counts.reset_index().rename(columns={"index":"category", col:"count"}), os.path.join(out, f"econ1_{label.lower()}_hist_W{w}.csv"))


def plot_econ2_distributions(weeks: List[int], matched_only: bool = False) -> None:
    # overall and per week
    name_overall = figure_name("ECON2_distribution_overall", matched_only)
    out_overall = figure_folder(name_overall)
    arr_llm = []
    arr_hum = []
    per_week_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        # Filter ECON1 eligible (1 or 2)
        mask_llm = df.get("ECON1_code_LLM").astype(str).isin(["1","2"]) if "ECON1_code_LLM" in df else pd.Series([False]*len(df))
        mask_hum = df.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in df else pd.Series([False]*len(df))
        econ2_llm = df.loc[mask_llm, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in df else np.array([])
        econ2_h = df.loc[mask_hum, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in df else np.array([])
        if len(econ2_llm) > 0:
            arr_llm.append(econ2_llm)
        if len(econ2_h) > 0:
            arr_hum.append(econ2_h)
        per_week_data[w] = (econ2_llm, econ2_h)

    if arr_llm and arr_hum:
        all_llm = np.concatenate(arr_llm)
        all_hum = np.concatenate(arr_hum)
        # overall histogram
        fig, ax = plt.subplots(figsize=(6,4))
        bins = np.arange(0, 169, 4)
        ax.hist(all_hum, bins=bins, alpha=0.6, label="Human", color="#55A868")
        ax.hist(all_llm, bins=bins, alpha=0.6, label="LLM", color="#C44E52")
        ax.set_title("ECON2 hours distribution (overall W1+W2)")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(out_overall, "econ2_distribution_overall.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        # Save arrays
        np.save(os.path.join(out_overall, "econ2_llm_overall.npy"), all_llm)
        np.save(os.path.join(out_overall, "econ2_human_overall.npy"), all_hum)

    # per week plots
    for w, (llm_vals, hum_vals) in per_week_data.items():
        name_w = figure_name(f"ECON2_distribution_W{w}", matched_only)
        out_w = figure_folder(name_w)
        if len(llm_vals) == 0 and len(hum_vals) == 0:
            continue
        fig, ax = plt.subplots(figsize=(6,4))
        bins = np.arange(0, 169, 4)
        if len(hum_vals) > 0:
            ax.hist(hum_vals, bins=bins, alpha=0.6, label="Human", color="#55A868")
        if len(llm_vals) > 0:
            ax.hist(llm_vals, bins=bins, alpha=0.6, label="LLM", color="#C44E52")
        ax.set_title(f"ECON2 hours distribution (W{w})")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(out_w, f"econ2_distribution_W{w}.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        # Save arrays
        np.save(os.path.join(out_w, f"econ2_llm_W{w}.npy"), llm_vals)
        np.save(os.path.join(out_w, f"econ2_human_W{w}.npy"), hum_vals)
        # Side-by-side comparison figure
        fig2, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
        if len(hum_vals) > 0:
            axes[0].hist(hum_vals, bins=bins, color="#55A868")
        axes[0].set_title(f"Human ECON2 (W{w})")
        axes[0].set_xlabel("Hours")
        axes[0].set_ylabel("Count")
        if len(llm_vals) > 0:
            axes[1].hist(llm_vals, bins=bins, color="#C44E52")
        axes[1].set_title(f"LLM ECON2 (W{w})")
        axes[1].set_xlabel("Hours")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_w, f"econ2_side_by_side_W{w}.png"), dpi=200)
        plt.close(fig2)


def compare_distributions(weeks: List[int], matched_only: bool = False) -> None:
    # KS and Wasserstein distances
    name_overall = figure_name("ECON2_llm_vs_human_distance_overall", matched_only)
    out_overall = figure_folder(name_overall)

    llm_all = []
    hum_all = []
    rows = []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        mask_llm = df.get("ECON1_code_LLM").astype(str).isin(["1","2"]) if "ECON1_code_LLM" in df else pd.Series([False]*len(df))
        mask_hum = df.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in df else pd.Series([False]*len(df))
        econ2_llm = df.loc[mask_llm, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in df else np.array([])
        econ2_h = df.loc[mask_hum, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in df else np.array([])
        if len(econ2_llm) and len(econ2_h):
            if SCIPY_AVAILABLE:
                ks = stats.ks_2samp(econ2_llm, econ2_h).statistic
                wdist = stats.wasserstein_distance(econ2_llm, econ2_h)
            else:
                # fallback: simple EMD via sorted absolute mean diff of CDFs approximated by percentiles
                ks = float(np.max(np.abs(np.linspace(0,1,len(econ2_llm)) - np.linspace(0,1,len(econ2_h)))))
                wdist = float(np.abs(np.mean(econ2_llm) - np.mean(econ2_h)))
            rows.append({"WEEK": w, "KS": ks, "Wasserstein": wdist, "LLM_N": int(len(econ2_llm)), "Human_N": int(len(econ2_h))})
            llm_all.append(econ2_llm)
            hum_all.append(econ2_h)
    df_metrics = pd.DataFrame(rows)
    if not df_metrics.empty:
        save_csv(df_metrics, os.path.join(out_overall, "econ2_distance_by_week.csv"))

    if llm_all and hum_all:
        all_llm = np.concatenate(llm_all)
        all_h = np.concatenate(hum_all)
        if SCIPY_AVAILABLE:
            ks = stats.ks_2samp(all_llm, all_h).statistic
            wdist = stats.wasserstein_distance(all_llm, all_h)
        else:
            ks = float(np.max(np.abs(np.linspace(0,1,len(all_llm)) - np.linspace(0,1,len(all_h)))))
            wdist = float(np.abs(np.mean(all_llm) - np.mean(all_h)))
        with open(os.path.join(out_overall, "econ2_distance_overall.json"), "w", encoding="utf-8") as f:
            json.dump({"KS": ks, "Wasserstein": wdist, "LLM_N": int(len(all_llm)), "Human_N": int(len(all_h))}, f, indent=2)


def pair_fit_econ2(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON2_pair_fit", matched_only)
    out = figure_folder(name)
    rows = []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        # pairs where both present and eligible (use human eligibility ECON1 in {1,2})
        if "ECON1_code" not in df or "ECON2_hours_Human" not in df or "ECON2_hours_LLM" not in df:
            continue
        mask_hum = df.get("ECON1_code").astype(str).isin(["1","2"]) 
        pairs = df.loc[mask_hum, ["SU_ID", "STATE", "ECON2_hours_Human", "ECON2_hours_LLM"]].copy()
        pairs = pairs.dropna(subset=["ECON2_hours_Human", "ECON2_hours_LLM"])
        if pairs.empty:
            continue
        pairs["WEEK"] = w
        rows.append(pairs)
    if not rows:
        return
    df_all = pd.concat(rows, ignore_index=True)
    # metrics
    y = df_all["ECON2_hours_LLM"].astype(float).values
    x = df_all["ECON2_hours_Human"].astype(float).values
    mae = float(np.mean(np.abs(y - x)))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    corr = float(np.corrcoef(x, y)[0,1]) if len(x) > 1 else np.nan
    with open(os.path.join(out, "pair_fit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"MAE": mae, "RMSE": rmse, "PearsonR": corr, "N": int(len(x))}, f, indent=2)
    # scatter
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x, y, alpha=0.3, s=10, color="#4C72B0")
    lim = (0, max(1, float(np.nanmax([x.max(), y.max()]))))
    ax.plot(lim, lim, linestyle='--', color='gray', label='y=x')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Human ECON2 hours")
    ax.set_ylabel("LLM ECON2 hours")
    ax.set_title("ECON2 LLM vs Human (pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "econ2_pair_scatter.png"), dpi=200)
    plt.close(fig)
    # save data
    save_csv(df_all, os.path.join(out, "econ2_pair_data.csv"))


def env_association(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON_env_association", matched_only)
    out = figure_folder(name)
    # Load environment csvs (structured)
    state_csv = os.path.join(PROJECT_ROOT, "data", "ny_tx_2020-04-10_to_2020-06-20_slim.csv")
    us_csv = os.path.join(PROJECT_ROOT, "data", "us_aggregate_2020-04-10_to_2020-06-20_slim.csv")
    st = pd.read_csv(state_csv)
    us = pd.read_csv(us_csv)
    st["Report_Date"] = pd.to_datetime(st["Report_Date"]) ; us["Report_Date"] = pd.to_datetime(us["Report_Date"]) 
    windows = {1: (pd.Timestamp("2020-04-12"), pd.Timestamp("2020-04-20")),
               2: (pd.Timestamp("2020-04-26"), pd.Timestamp("2020-05-04")),
               3: (pd.Timestamp("2020-05-24"), pd.Timestamp("2020-05-30"))}
    rows = []
    for w in weeks:
        if w not in windows:
            continue
        start, end = windows[w]
        for state in ["New York", "Texas"]:
            swin = st[(st["Province_State"]==state) & (st["Report_Date"]>=start) & (st["Report_Date"]<=end)].copy()
            if swin.empty:
                continue
            # daily diffs
            swin = swin.sort_values("Report_Date")
            new_cases = swin["Confirmed"].diff().dropna()
            new_deaths = swin["Deaths"].diff().dropna()
            feat = {
                "WEEK": w,
                "STATE_full": state,
                "STATE": "NY" if state=="New York" else "TX",
                "avg_new_cases": float(new_cases.mean()) if len(new_cases)>0 else np.nan,
                "avg_new_deaths": float(new_deaths.mean()) if len(new_deaths)>0 else np.nan,
                "avg_hospitalized": float(swin["People_Hospitalized"].dropna().astype(float).mean()) if "People_Hospitalized" in swin else np.nan,
            }
            rows.append(feat)
    env_df = pd.DataFrame(rows)
    if env_df.empty:
        return
    save_csv(env_df, os.path.join(out, "env_features.csv"))

    # Aggregate ECON1 rates and ECON2 means by state-week (LLM/Human)
    agg_rows = []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        for state in ["NY","TX"]:
            sub = df[df["STATE"]==state]
            if sub.empty:
                continue
            # ECON1 rate (Yes employee code=1)
            r_llm = np.mean((sub.get("ECON1_code_LLM").astype(str)=="1").values) if "ECON1_code_LLM" in sub else np.nan
            r_hum = np.mean((sub.get("ECON1_code").astype(str)=="1").values)
            # ECON2 means
            m_llm = float(sub.loc[sub.get("ECON1_code_LLM").astype(str)=="1", "ECON2_hours_LLM"].dropna().astype(float).mean()) if ("ECON1_code_LLM" in sub and "ECON2_hours_LLM" in sub) else np.nan
            def human_hours_series(s: pd.Series):
                return s.apply(lambda x: float(re.match(r"\s*(\d+)", x).group(1)) if isinstance(x,str) and re.match(r"\s*(\d+)", x) else np.nan)
            if ("ECON1_code" in sub and "ECON2" in sub):
                m_hum = float(human_hours_series(sub.loc[sub.get("ECON1_code").astype(str)=="1", "ECON2"]).dropna().mean())
            else:
                m_hum = np.nan
            agg_rows.append({"WEEK": w, "STATE": state, "econ1_rate_llm": r_llm, "econ1_rate_human": r_hum, "econ2_mean_llm": m_llm, "econ2_mean_human": m_hum})
    agg_df = pd.DataFrame(agg_rows)
    save_csv(agg_df, os.path.join(out, "econ_state_week_aggregates.csv"))

    # Join and plot bar charts
    join = agg_df.merge(env_df, on=["WEEK","STATE"], how="left")
    save_csv(join, os.path.join(out, "econ_env_join.csv"))

    # Plot ECON2 means by state-week (dual series)
    fig, ax = plt.subplots(figsize=(7,4))
    xlabels = []
    series_h = []
    series_l = []
    for w in weeks:
        for st in ["NY","TX"]:
            row = join[(join["WEEK"]==w) & (join["STATE"]==st)]
            if not row.empty:
                series_h.append(float(row["econ2_mean_human"].values[0]) if "econ2_mean_human" in row else np.nan)
                series_l.append(float(row["econ2_mean_llm"].values[0]) if "econ2_mean_llm" in row else np.nan)
                xlabels.append(f"W{w}-{st}")
    ax.plot(xlabels, series_h, marker='o', label='econ2_mean_human', color='#55A868')
    ax.plot(xlabels, series_l, marker='o', label='econ2_mean_llm', color='#C44E52')
    ax.set_title("ECON2 mean by state-week (Human vs LLM)")
    ax.set_ylabel("Hours")
    ax.set_xlabel("State-Week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "econ2_means_by_state_week.png"), dpi=200)
    plt.close(fig)

    # Pearson correlation between env metrics and targets at state-week level
    corr_rows = []
    for metric in ["avg_new_cases", "avg_new_deaths", "avg_hospitalized"]:
        for col in ["econ2_mean_human", "econ2_mean_llm", "econ1_rate_human", "econ1_rate_llm"]:
            if metric in join.columns and col in join.columns:
                a = pd.to_numeric(join[metric], errors='coerce')
                b = pd.to_numeric(join[col], errors='coerce')
                if a.notna().sum() > 1 and b.notna().sum() > 1:
                    r = float(np.corrcoef(a.fillna(0), b.fillna(0))[0,1])
                else:
                    r = np.nan
                corr_rows.append({"metric": metric, "target": col, "pearson_r": r})
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        save_csv(corr_df, os.path.join(out, "env_correlations.csv"))


def econ2_tail_shares(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON2_tail_shares", matched_only)
    out = figure_folder(name)
    thresholds = [60, 80, 100]
    rows = []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        mask_h = df.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in df else pd.Series([False]*len(df))
        mask_l = df.get("ECON1_code_LLM").astype(str).isin(["1","2"]) if "ECON1_code_LLM" in df else pd.Series([False]*len(df))
        h = df.loc[mask_h, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in df else np.array([])
        l = df.loc[mask_l, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in df else np.array([])
        for t in thresholds:
            def share(arr):
                return float(np.mean(arr > t)) if len(arr)>0 else np.nan
            rows.append({"WEEK": w, "threshold": t, "human_share": share(h), "llm_share": share(l), "human_n": int(len(h)), "llm_n": int(len(l))})
        # Plot per week bars
        if len(h)>0 or len(l)>0:
            fig, ax = plt.subplots(figsize=(6,4))
            x = np.arange(len(thresholds))
            width = 0.38
            human_vals = [float(np.mean(h>t)) if len(h)>0 else np.nan for t in thresholds]
            llm_vals = [float(np.mean(l>t)) if len(l)>0 else np.nan for t in thresholds]
            ax.bar(x - width/2, human_vals, width, label="Human", color="#55A868")
            ax.bar(x + width/2, llm_vals, width, label="LLM", color="#C44E52")
            ax.set_xticks(x)
            ax.set_xticklabels([f">{t}h" for t in thresholds])
            ax.set_ylim(0,1)
            ax.set_ylabel("Share")
            ax.set_title(f"ECON2 tail shares (W{w})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"econ2_tail_W{w}.png"), dpi=200)
            plt.close(fig)
    if rows:
        save_csv(pd.DataFrame(rows), os.path.join(out, "econ2_tail_shares.csv"))


def econ2_qq_plots(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON2_QQ", matched_only)
    out = figure_folder(name)
    def qq_plot(x: np.ndarray, y: np.ndarray, title: str, path: str) -> None:
        if len(x)==0 or len(y)==0:
            return
        qs = np.linspace(0.01, 0.99, 99)
        xq = np.quantile(x, qs)
        yq = np.quantile(y, qs)
        lim = (0, max(float(np.nanmax([xq.max(), yq.max()])), 1.0))
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(xq, yq, s=12, alpha=0.7, color="#4C72B0")
        ax.plot(lim, lim, linestyle='--', color='gray')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Human quantiles")
        ax.set_ylabel("LLM quantiles")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
    # overall
    arr_h, arr_l = [], []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        mask_h = df.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in df else pd.Series([False]*len(df))
        mask_l = df.get("ECON1_code_LLM").astype(str).isin(["1","2"]) if "ECON1_code_LLM" in df else pd.Series([False]*len(df))
        if "ECON2_hours_Human" in df:
            arr_h.append(df.loc[mask_h, "ECON2_hours_Human"].dropna().astype(float).values)
        if "ECON2_hours_LLM" in df:
            arr_l.append(df.loc[mask_l, "ECON2_hours_LLM"].dropna().astype(float).values)
        # per week
        h = arr_h[-1] if arr_h else np.array([])
        l = arr_l[-1] if arr_l else np.array([])
        qq_plot(h, l, f"ECON2 QQ (W{w})", os.path.join(out, f"econ2_qq_W{w}.png"))
    if arr_h and arr_l:
        h_all = np.concatenate(arr_h)
        l_all = np.concatenate(arr_l)
        qq_plot(h_all, l_all, "ECON2 QQ (overall)", os.path.join(out, "econ2_qq_overall.png"))


def write_coverage_report(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("coverage_report", matched_only)
    out = figure_folder(name)
    rows = []
    for w in weeks:
        human = load_human_week(w)
        llm = load_llm_week(w)
        rows.append({
            "WEEK": w,
            "human_rows": int(len(human)) if not human.empty else 0,
            "llm_rows": int(len(llm)) if not llm.empty else 0,
            "llm_econ1_present": int(llm["ECON1_code_LLM"].notna().sum()) if (not llm.empty and "ECON1_code_LLM" in llm) else 0,
            "llm_econ2_present": int(llm["ECON2"].notna().sum()) if (not llm.empty and "ECON2" in llm) else 0,
        })
    if rows:
        save_csv(pd.DataFrame(rows), os.path.join(out, "coverage_summary.csv"))


def plot_hist_econ1_by_state(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON1_hist_by_state", matched_only)
    out = figure_folder(name)
    econ1_map = {"1": "Yes employee", "2": "Self-employed", "3": "No", "": "Unknown"}
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty or "STATE" not in df.columns:
            continue
        for st in ["NY", "TX"]:
            sub = df[df["STATE"] == st].copy()
            if sub.empty:
                continue
            sub["ECON1_code_LLM_lbl"] = sub.get("ECON1_code_LLM").fillna("").astype(str).map(econ1_map).fillna("Other")
            sub["ECON1_code_human_lbl"] = sub.get("ECON1_code").fillna("").astype(str).map(econ1_map).fillna("Other")
            for label, col in [("LLM", "ECON1_code_LLM_lbl"), ("Human", "ECON1_code_human_lbl")]:
                fig, ax = plt.subplots(figsize=(6,4))
                counts = sub[col].value_counts().reindex(["Yes employee", "Self-employed", "No", "Unknown", "Other"], fill_value=0)
                counts.plot(kind='bar', ax=ax, color="#4C72B0")
                ax.set_title(f"ECON1 {label} histogram (W{w}, {st})")
                ax.set_ylabel("Count")
                ax.set_xlabel("Category")
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"econ1_{label.lower()}_hist_W{w}_{st}.png"), dpi=200)
                plt.close(fig)
                save_csv(counts.reset_index().rename(columns={"index":"category", col:"count"}), os.path.join(out, f"econ1_{label.lower()}_hist_W{w}_{st}.csv"))


def plot_econ2_distributions_by_state(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON2_distribution_by_state", matched_only)
    out_base = figure_folder(name)
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty or "STATE" not in df.columns:
            continue
        bins = np.arange(0, 169, 4)
        for st in ["NY", "TX"]:
            sub = df[df["STATE"] == st].copy()
            if sub.empty:
                continue
            mask_h = sub.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in sub else pd.Series([False]*len(sub))
            mask_l = sub.get("ECON1_code_LLM").astype(str).isin(["1","2"]) if "ECON1_code_LLM" in sub else pd.Series([False]*len(sub))
            h = sub.loc[mask_h, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in sub else np.array([])
            l = sub.loc[mask_l, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in sub else np.array([])
            if len(h)==0 and len(l)==0:
                continue
            # overlay
            fig, ax = plt.subplots(figsize=(6,4))
            if len(h)>0:
                ax.hist(h, bins=bins, alpha=0.6, label="Human", color="#55A868")
            if len(l)>0:
                ax.hist(l, bins=bins, alpha=0.6, label="LLM", color="#C44E52")
            ax.set_title(f"ECON2 hours distribution (W{w}, {st})")
            ax.set_xlabel("Hours")
            ax.set_ylabel("Count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_base, f"econ2_distribution_W{w}_{st}.png"), dpi=200)
            plt.close(fig)
            # side-by-side
            fig2, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
            if len(h)>0:
                axes[0].hist(h, bins=bins, color="#55A868")
            axes[0].set_title(f"Human ECON2 (W{w}, {st})")
            axes[0].set_xlabel("Hours")
            axes[0].set_ylabel("Count")
            if len(l)>0:
                axes[1].hist(l, bins=bins, color="#C44E52")
            axes[1].set_title(f"LLM ECON2 (W{w}, {st})")
            axes[1].set_xlabel("Hours")
            fig2.tight_layout()
            fig2.savefig(os.path.join(out_base, f"econ2_side_by_side_W{w}_{st}.png"), dpi=200)
            plt.close(fig2)


def pair_fit_econ2_by_state(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON2_pair_fit_by_state", matched_only)
    out = figure_folder(name)
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty or "STATE" not in df.columns:
            continue
        for st in ["NY", "TX"]:
            sub = df[df["STATE"]==st].copy()
            if sub.empty or "ECON2_hours_LLM" not in sub or "ECON2_hours_Human" not in sub:
                continue
            mask_hum = sub.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in sub else pd.Series([False]*len(sub))
            pairs = sub.loc[mask_hum, ["ECON2_hours_Human", "ECON2_hours_LLM"]].dropna()
            if pairs.empty:
                continue
            y = pairs["ECON2_hours_LLM"].astype(float).values
            x = pairs["ECON2_hours_Human"].astype(float).values
            mae = float(np.mean(np.abs(y - x)))
            rmse = float(np.sqrt(np.mean((y - x) ** 2)))
            corr = float(np.corrcoef(x, y)[0,1]) if len(x) > 1 else np.nan
            with open(os.path.join(out, f"pair_fit_W{w}_{st}.json"), "w", encoding="utf-8") as f:
                json.dump({"MAE": mae, "RMSE": rmse, "PearsonR": corr, "N": int(len(x))}, f, indent=2)
            fig, ax = plt.subplots(figsize=(5,5))
            ax.scatter(x, y, alpha=0.3, s=10, color="#4C72B0")
            lim = (0, max(1, float(np.nanmax([x.max(), y.max()]))))
            ax.plot(lim, lim, linestyle='--', color='gray', label='y=x')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("Human ECON2 hours")
            ax.set_ylabel("LLM ECON2 hours")
            ax.set_title(f"ECON2 LLM vs Human (W{w}, {st})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"econ2_pair_scatter_W{w}_{st}.png"), dpi=200)
            plt.close(fig)


def econ2_tail_shares_by_state(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON2_tail_shares_by_state", matched_only)
    out = figure_folder(name)
    thresholds = [60, 80, 100]
    rows = []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty or "STATE" not in df.columns:
            continue
        for st in ["NY", "TX"]:
            sub = df[df["STATE"]==st].copy()
            if sub.empty:
                continue
            mask_h = sub.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in sub else pd.Series([False]*len(sub))
            mask_l = sub.get("ECON1_code_LLM").astype(str).isin(["1","2"]) if "ECON1_code_LLM" in sub else pd.Series([False]*len(sub))
            h = sub.loc[mask_h, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in sub else np.array([])
            l = sub.loc[mask_l, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in sub else np.array([])
            for t in thresholds:
                def share(arr):
                    return float(np.mean(arr > t)) if len(arr)>0 else np.nan
                rows.append({"WEEK": w, "STATE": st, "threshold": t, "human_share": share(h), "llm_share": share(l), "human_n": int(len(h)), "llm_n": int(len(l))})
            # plot bars per state-week
            if len(h)>0 or len(l)>0:
                fig, ax = plt.subplots(figsize=(6,4))
                x = np.arange(len(thresholds))
                width = 0.38
                human_vals = [float(np.mean(h>t)) if len(h)>0 else np.nan for t in thresholds]
                llm_vals = [float(np.mean(l>t)) if len(l)>0 else np.nan for t in thresholds]
                ax.bar(x - width/2, human_vals, width, label="Human", color="#55A868")
                ax.bar(x + width/2, llm_vals, width, label="LLM", color="#C44E52")
                ax.set_xticks(x)
                ax.set_xticklabels([f">{t}h" for t in thresholds])
                ax.set_ylim(0,1)
                ax.set_ylabel("Share")
                ax.set_title(f"ECON2 tail shares (W{w}, {st})")
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"econ2_tail_W{w}_{st}.png"), dpi=200)
                plt.close(fig)
    if rows:
        save_csv(pd.DataFrame(rows), os.path.join(out, "econ2_tail_shares_by_state.csv"))


def econ2_qq_plots_by_state(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON2_QQ_by_state", matched_only)
    out = figure_folder(name)
    def qq_plot(x: np.ndarray, y: np.ndarray, title: str, path: str) -> None:
        if len(x)==0 or len(y)==0:
            return
        qs = np.linspace(0.01, 0.99, 99)
        xq = np.quantile(x, qs)
        yq = np.quantile(y, qs)
        lim = (0, max(float(np.nanmax([xq.max(), yq.max()])), 1.0))
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(xq, yq, s=12, alpha=0.7, color="#4C72B0")
        ax.plot(lim, lim, linestyle='--', color='gray')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Human quantiles")
        ax.set_ylabel("LLM quantiles")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty or "STATE" not in df.columns:
            continue
        for st in ["NY", "TX"]:
            sub = df[df["STATE"]==st].copy()
            if sub.empty:
                continue
            mask_h = sub.get("ECON1_code").astype(str).isin(["1","2"]) if "ECON1_code" in sub else pd.Series([False]*len(sub))
            mask_l = sub.get("ECON1_code_LLM").astype(str).isin(["1","2"]) if "ECON1_code_LLM" in sub else pd.Series([False]*len(sub))
            h = sub.loc[mask_h, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in sub else np.array([])
            l = sub.loc[mask_l, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in sub else np.array([])
            qq_plot(h, l, f"ECON2 QQ (W{w}, {st})", os.path.join(out, f"econ2_qq_W{w}_{st}.png"))


def persona_segment_econ1_closeness(weeks: List[int], matched_only: bool = False) -> None:
    name = figure_name("ECON1_persona_segment_closeness", matched_only)
    out = figure_folder(name)
    seg_rows = []
    for w in weeks:
        df = merge_llm_human_matched(w, matched_only)
        if df.empty:
            continue
        # choose a few persona vars if present
        candidates = [
            ("PHYS8", "General health"),
            ("SOC5A", "Anxious (7 days)"),
            ("P_DENSE", "Population density"),
        ]
        for seg_col, title in candidates:
            if seg_col not in df.columns:
                continue
            seg = df[["STATE", "SU_ID", seg_col, "ECON1_code", "ECON1_code_LLM"]].copy()
            seg["segment"] = seg[seg_col].astype(str)
            g = seg.groupby("segment")
            rates = []
            for k, s in g:
                r_h = np.mean(s.get("ECON1_code").astype(str).isin(["1","2"]).values) if "ECON1_code" in s else np.nan
                r_l = np.mean(s.get("ECON1_code_LLM").astype(str).isin(["1","2"]).values) if "ECON1_code_LLM" in s else np.nan
                n = len(s)
                rates.append({"WEEK": w, "segment": k, "rate_human": r_h, "rate_llm": r_l, "N": n, "seg_col": seg_col})
            seg_rates = pd.DataFrame(rates).sort_values("segment")
            if not seg_rates.empty:
                save_csv(seg_rates, os.path.join(out, f"rates_W{w}_{seg_col}.csv"))
                top = seg_rates.sort_values("N", ascending=False).head(8)
                x = np.arange(len(top))
                width = 0.38
                fig, ax = plt.subplots(figsize=(10,4))
                ax.bar(x - width/2, top["rate_human"].values, width, label="Human", color="#55A868")
                ax.bar(x + width/2, top["rate_llm"].values, width, label="LLM", color="#C44E52")
                ax.set_xticks(x)
                ax.set_xticklabels(top["segment"].astype(str).tolist(), rotation=30, ha='right')
                ax.set_ylim(0,1)
                ax.set_ylabel("Rate ECON1==1")
                ax.set_title(f"ECON1 working rate by persona segment (W{w}, {title})")
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"econ1_rate_W{w}_{seg_col}.png"), dpi=200)
                plt.close(fig)


def brainstorm_notes() -> None:
    notes = {
        "ideas": [
            "Quantile comparison for ECON2 (QQ plots) to inspect tail behavior.",
            "Conditional distributions of ECON2 by persona segments (e.g., PHYS8 health status).",
            "Robustness: sensitivity of ECON2 to env windows (shift by +/-2 days).",
            "Bootstrap CIs for distribution distances (KS/Wasserstein).",
            "Propensity to report any hours (ECON1==1) vs env severity (rate differences across states/weeks).",
            "Heterogeneity: compare NY vs TX separately across weeks for LLM vs Human gaps.",
            "Outlier analysis on ECON2 (e.g., >100 hours) shares, LLM vs Human.",
        ]
    }
    out = figure_folder("BRAINSTORM_NOTES")
    with open(os.path.join(out, "notes.json"), "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=str, default="1,2,3", help="Comma-separated weeks, e.g., 1,2,3")
    parser.add_argument("--matched_only", action="store_true", help="Use only samples with LLM answers (matched subset)")
    args = parser.parse_args()

    weeks = [int(w.strip()) for w in args.weeks.split(",") if w.strip()]
    plot_hist_econ1(weeks, matched_only=args.matched_only)
    plot_hist_econ1_per_week(weeks, matched_only=args.matched_only)
    plot_econ2_distributions(weeks, matched_only=args.matched_only)
    plot_econ2_distributions_by_state(weeks, matched_only=args.matched_only)
    compare_distributions(weeks, matched_only=args.matched_only)
    pair_fit_econ2(weeks, matched_only=args.matched_only)
    pair_fit_econ2_by_state(weeks, matched_only=args.matched_only)
    env_association(weeks, matched_only=args.matched_only)
    persona_segment_econ1_closeness(weeks, matched_only=args.matched_only)
    econ2_tail_shares(weeks, matched_only=args.matched_only)
    econ2_tail_shares_by_state(weeks, matched_only=args.matched_only)
    econ2_qq_plots(weeks, matched_only=args.matched_only)
    econ2_qq_plots_by_state(weeks, matched_only=args.matched_only)
    write_coverage_report(weeks, matched_only=args.matched_only)
    brainstorm_notes()
    print("Inference completed. See outputs/diagrams/* for artifacts.")


if __name__ == "__main__":
    main()
