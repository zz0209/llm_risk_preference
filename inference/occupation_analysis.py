import os
import re
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse merge helper from the main inference script (support both package and script runs)
try:
    from .run_inference import merge_llm_human_matched
except Exception:
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from run_inference import merge_llm_human_matched


# ---------------------- Helpers ----------------------
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def slugify(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^0-9A-Za-z_\-]+", "_", value)
    return value[:120]


def is_valid_occupation(val: object) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    s = str(val).strip()
    if s == "" or s.upper() in {"NA", "N/A", "NONE"}:
        return False
    return True


def econ1_label_map() -> Dict[str, str]:
    return {"1": "Yes employee", "2": "Self-employed", "3": "No", "": "Unknown"}


def load_all_weeks(matched_only: bool = False) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for w in [1, 2, 3]:
        try:
            dfw = merge_llm_human_matched(w, matched_only)
            if dfw is None or dfw.empty:
                continue
            dfw = dfw.copy()
            dfw["WEEK"] = w
            frames.append(dfw)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def plot_econ1_by_occupation(df: pd.DataFrame, out_dir: str) -> None:
    label_map = econ1_label_map()
    categories = (
        df.loc[df["P_OCCUPY2"].apply(is_valid_occupation), "P_OCCUPY2"].value_counts().index.tolist()
    )
    for occ in categories:
        sub = df[df["P_OCCUPY2"] == occ].copy()
        if sub.empty:
            continue
        sub["ECON1_code_LLM_lbl"] = sub.get("ECON1_code_LLM").fillna("").astype(str).map(label_map).fillna("Other")
        sub["ECON1_code_human_lbl"] = sub.get("ECON1_code").fillna("").astype(str).map(label_map).fillna("Other")

        # Side-by-side bar counts
        order = ["Yes employee", "Self-employed", "No", "Unknown", "Other"]
        human_counts = sub["ECON1_code_human_lbl"].value_counts().reindex(order, fill_value=0)
        llm_counts = sub["ECON1_code_LLM_lbl"].value_counts().reindex(order, fill_value=0)
        idx = np.arange(len(order))
        width = 0.42
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(idx - width/2, human_counts.values, width, label="Human", color="#55A868")
        ax.bar(idx + width/2, llm_counts.values, width, label="LLM", color="#C44E52")
        ax.set_xticks(idx)
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylabel("Count")
        ax.set_title(f"ECON1 distribution by occupation: {occ}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"econ1_bar_{slugify(occ)}.png"), dpi=200)
        plt.close(fig)


def plot_econ2_by_occupation(df: pd.DataFrame, out_dir: str) -> None:
    categories = (
        df.loc[df["P_OCCUPY2"].apply(is_valid_occupation), "P_OCCUPY2"].value_counts().index.tolist()
    )
    bins = np.arange(0, 169, 4)
    for occ in categories:
        sub = df[df["P_OCCUPY2"] == occ].copy()
        if sub.empty:
            continue
        mask_h = sub.get("ECON1_code").astype(str).isin(["1", "2"]) if "ECON1_code" in sub else pd.Series([False] * len(sub))
        mask_l = sub.get("ECON1_code_LLM").astype(str).isin(["1", "2"]) if "ECON1_code_LLM" in sub else pd.Series([False] * len(sub))
        h = sub.loc[mask_h, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in sub else np.array([])
        l = sub.loc[mask_l, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in sub else np.array([])
        if len(h) == 0 and len(l) == 0:
            continue

        # Overlay
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(h) > 0:
            ax.hist(h, bins=bins, alpha=0.6, label="Human", color="#55A868")
        if len(l) > 0:
            ax.hist(l, bins=bins, alpha=0.6, label="LLM", color="#C44E52")
        ax.set_title(f"ECON2 hours distribution by occupation: {occ}")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"econ2_overlay_{slugify(occ)}.png"), dpi=200)
        plt.close(fig)

        # Side-by-side
        fig2, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        if len(h) > 0:
            axes[0].hist(h, bins=bins, color="#55A868")
        axes[0].set_title(f"Human ECON2: {occ}")
        axes[0].set_xlabel("Hours")
        axes[0].set_ylabel("Count")
        if len(l) > 0:
            axes[1].hist(l, bins=bins, color="#C44E52")
        axes[1].set_title(f"LLM ECON2: {occ}")
        axes[1].set_xlabel("Hours")
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"econ2_side_by_side_{slugify(occ)}.png"), dpi=200)
        plt.close(fig2)


def pair_fit_econ2_by_occupation(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    rows = []
    categories = (
        df.loc[df["P_OCCUPY2"].apply(is_valid_occupation), "P_OCCUPY2"].value_counts().index.tolist()
    )
    for occ in categories:
        sub = df[df["P_OCCUPY2"] == occ].copy()
        if sub.empty or "ECON2_hours_LLM" not in sub or "ECON2_hours_Human" not in sub:
            continue
        mask_hum = sub.get("ECON1_code").astype(str).isin(["1", "2"]) if "ECON1_code" in sub else pd.Series([False] * len(sub))
        pairs = sub.loc[mask_hum, ["ECON2_hours_Human", "ECON2_hours_LLM"]].dropna()
        if pairs.empty:
            continue
        y = pairs["ECON2_hours_LLM"].astype(float).values
        x = pairs["ECON2_hours_Human"].astype(float).values
        mae = float(np.mean(np.abs(y - x)))
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
        rows.append({
            "occupation": occ,
            "N_pairs": int(len(x)),
            "MAE": mae,
            "RMSE": rmse,
            "PearsonR": corr,
        })
        # scatter
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x, y, alpha=0.3, s=12, color="#4C72B0")
        lim = (0, max(1.0, float(np.nanmax([x.max(), y.max()]))))
        ax.plot(lim, lim, linestyle='--', color='gray', label='y=x')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Human ECON2 hours")
        ax.set_ylabel("LLM ECON2 hours")
        ax.set_title(f"ECON2 LLM vs Human (occupation: {occ})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"econ2_pair_scatter_{slugify(occ)}.png"), dpi=200)
        plt.close(fig)
    return pd.DataFrame(rows)


def econ2_tail_by_occupation(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    thresholds = [60, 80, 100]
    rows = []
    categories = (
        df.loc[df["P_OCCUPY2"].apply(is_valid_occupation), "P_OCCUPY2"].value_counts().index.tolist()
    )
    for occ in categories:
        sub = df[df["P_OCCUPY2"] == occ].copy()
        if sub.empty:
            continue
        mask_h = sub.get("ECON1_code").astype(str).isin(["1", "2"]) if "ECON1_code" in sub else pd.Series([False] * len(sub))
        mask_l = sub.get("ECON1_code_LLM").astype(str).isin(["1", "2"]) if "ECON1_code_LLM" in sub else pd.Series([False] * len(sub))
        h = sub.loc[mask_h, "ECON2_hours_Human"].dropna().astype(float).values if "ECON2_hours_Human" in sub else np.array([])
        l = sub.loc[mask_l, "ECON2_hours_LLM"].dropna().astype(float).values if "ECON2_hours_LLM" in sub else np.array([])
        if len(h) == 0 and len(l) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(thresholds))
        width = 0.38
        human_vals = [float(np.mean(h > t)) if len(h) > 0 else np.nan for t in thresholds]
        llm_vals = [float(np.mean(l > t)) if len(l) > 0 else np.nan for t in thresholds]
        ax.bar(x - width/2, human_vals, width, label="Human", color="#55A868")
        ax.bar(x + width/2, llm_vals, width, label="LLM", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels([f">{t}h" for t in thresholds])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Share")
        ax.set_title(f"ECON2 tail shares (occupation: {occ})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"econ2_tail_{slugify(occ)}.png"), dpi=200)
        plt.close(fig)
        for t, hv, lv in zip(thresholds, human_vals, llm_vals):
            rows.append({
                "occupation": occ,
                "threshold": t,
                "human_share": hv,
                "llm_share": lv,
                "human_n": int(len(h)),
                "llm_n": int(len(l)),
            })
    return pd.DataFrame(rows)


def econ1_rate_by_occupation(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    rows = []
    categories = (
        df.loc[df["P_OCCUPY2"].apply(is_valid_occupation), "P_OCCUPY2"].value_counts().index.tolist()
    )
    for occ in categories:
        sub = df[df["P_OCCUPY2"] == occ].copy()
        if sub.empty:
            continue
        def rate(col: str, code: str) -> float:
            if col not in sub:
                return np.nan
            denom = sub[col].notna().sum()
            if denom == 0:
                return np.nan
            return float((sub[col].astype(str) == code).mean())
        rows.append({
            "occupation": occ,
            "econ1_rate_human_yes_emp": rate("ECON1_code", "1"),
            "econ1_rate_human_self": rate("ECON1_code", "2"),
            "econ1_rate_human_no": rate("ECON1_code", "3"),
            "econ1_rate_llm_yes_emp": rate("ECON1_code_LLM", "1"),
            "econ1_rate_llm_self": rate("ECON1_code_LLM", "2"),
            "econ1_rate_llm_no": rate("ECON1_code_LLM", "3"),
            "N": int(len(sub)),
        })
        # Plot grouped bars for 3 categories
        cats = ["Yes employee", "Self-employed", "No"]
        human_vals = [rows[-1]["econ1_rate_human_yes_emp"], rows[-1]["econ1_rate_human_self"], rows[-1]["econ1_rate_human_no"]]
        llm_vals = [rows[-1]["econ1_rate_llm_yes_emp"], rows[-1]["econ1_rate_llm_self"], rows[-1]["econ1_rate_llm_no"]]
        x = np.arange(len(cats))
        width = 0.38
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width/2, human_vals, width, label="Human", color="#55A868")
        ax.bar(x + width/2, llm_vals, width, label="LLM", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
        ax.set_title(f"ECON1 category rates (occupation: {occ})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"econ1_rates_{slugify(occ)}.png"), dpi=200)
        plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    out_root = ensure_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "diagrams", "occupation"))
    # Merge all weeks, not matched-only by default as per instruction
    df = load_all_weeks(matched_only=False)
    if df.empty or "P_OCCUPY2" not in df.columns:
        print("No data or occupation column not found. Exiting.")
        return

    # Keep only rows with valid occupation
    df_occ = df[df["P_OCCUPY2"].apply(is_valid_occupation)].copy()
    if df_occ.empty:
        print("No non-NA occupation rows found. Exiting.")
        return

    # ECON1 distribution (bars) per occupation
    plot_econ1_by_occupation(df_occ, out_root)
    # ECON2 distributions (overlay + side-by-side) per occupation
    plot_econ2_by_occupation(df_occ, out_root)
    # ECON2 paired fit per occupation
    pair_df = pair_fit_econ2_by_occupation(df_occ, out_root)
    if not pair_df.empty:
        save_csv(pair_df, os.path.join(out_root, "econ2_pair_metrics_by_occupation.csv"))

    # Tail shares per occupation
    tail_df = econ2_tail_by_occupation(df_occ, out_root)
    if not tail_df.empty:
        save_csv(tail_df, os.path.join(out_root, "econ2_tail_shares_by_occupation.csv"))

    # ECON1 rates comparison per occupation
    rates_df = econ1_rate_by_occupation(df_occ, out_root)
    if not rates_df.empty:
        save_csv(rates_df, os.path.join(out_root, "econ1_rates_by_occupation.csv"))

    print("Occupation analysis completed. See outputs/diagrams/occupation/")


if __name__ == "__main__":
    main()


