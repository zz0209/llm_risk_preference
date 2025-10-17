import os
import re
import json
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def iter_outputs(root: str) -> List[Dict]:
    rows = []
    if not os.path.isdir(root):
        return rows
    pat = re.compile(r"^([A-Z]{2})_W(\d)_([0-9]+)_(ECON1|ECON2|PERCEPTION)\.txt$")
    for wk in (1, 2, 3):
        wk_dir = os.path.join(root, f"week{wk}")
        if not os.path.isdir(wk_dir):
            continue
        for fn in os.listdir(wk_dir):
            if not fn.endswith(".txt"):
                continue
            m = pat.match(fn)
            if not m:
                continue
            state, w, su, kind = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            content = read_text(os.path.join(wk_dir, fn))
            rows.append({"STATE": state, "WEEK": w, "SU_ID": su, "KIND": kind, "content": content})
    return rows


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def parse_econ1_worked(s: str) -> float:
    if not isinstance(s, str):
        return np.nan
    m = re.search(r"\((\d+)\)", s)
    if not m:
        return np.nan
    code = m.group(1)
    if code in {"1", "2"}:
        return 1.0
    if code == "3":
        return 0.0
    return np.nan


def parse_econ2_hours(s: str) -> float:
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


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_root = os.path.join(base, "outputs", "ECON_perception_heuritics")
    out_root = os.path.join(base, "outputs", "diagrams", "perception_heuritics_explainability")
    ensure_dir(out_root)

    rows = iter_outputs(in_root)
    # Build pooled dataset
    items = {}
    for r in rows:
        key = (r["STATE"], r["WEEK"], r["SU_ID"])  # keep week for potential future, but pooled analysis ignores it
        if key not in items:
            items[key] = {"STATE": r["STATE"], "WEEK": r["WEEK"], "SU_ID": r["SU_ID"],
                          "perception": np.nan, "econ1": np.nan, "econ2": np.nan}
        if r["KIND"] == "PERCEPTION":
            try:
                v = float(r["content"])
                if 0.0 <= v <= 1.0:
                    items[key]["perception"] = v
            except Exception:
                pass
        elif r["KIND"] == "ECON1":
            items[key]["econ1"] = parse_econ1_worked(r["content"])  # 1 if worked, 0 if not
        elif r["KIND"] == "ECON2":
            items[key]["econ2"] = parse_econ2_hours(r["content"])  # hours

    # Convert to arrays (pooled)
    P = []
    Y1 = []  # worked
    Y2 = []  # hours
    ST = []
    for k, it in items.items():
        if not np.isnan(it["perception"]):
            ST.append(it["STATE"])
            P.append(it["perception"])  # always record for denominators
            Y1.append(it["econ1"])     # may be NaN
            Y2.append(it["econ2"])     # may be NaN
    P = np.array(P, dtype=float)
    Y1 = np.array(Y1, dtype=float)
    Y2 = np.array(Y2, dtype=float)
    ST = np.array(ST, dtype=object)

    # ECON1 explainability (pooled)
    mask1 = ~np.isnan(Y1)
    econ1_corr = float(np.corrcoef(P[mask1], Y1[mask1])[0,1]) if (mask1.sum()>=2 and np.std(P[mask1])>0 and np.std(Y1[mask1])>0) else np.nan
    # OLS R2 for econ1
    econ1_r2 = np.nan
    if mask1.sum()>=2:
        X = np.vstack([np.ones(mask1.sum()), P[mask1]]).T
        y = Y1[mask1]
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            yhat = X @ beta
            ss_res = float(np.sum((y - yhat)**2))
            ss_tot = float(np.sum((y - np.mean(y))**2))
            econ1_r2 = float(1 - ss_res/ss_tot) if ss_tot>0 else np.nan
        except Exception:
            pass

    # ECON2 explainability (pooled, only where hours present)
    mask2 = ~np.isnan(Y2)
    econ2_corr = float(np.corrcoef(P[mask2], Y2[mask2])[0,1]) if (mask2.sum()>=2 and np.std(P[mask2])>0 and np.std(Y2[mask2])>0) else np.nan
    econ2_r2 = np.nan
    if mask2.sum()>=2:
        X = np.vstack([np.ones(mask2.sum()), P[mask2]]).T
        y = Y2[mask2]
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            yhat = X @ beta
            ss_res = float(np.sum((y - yhat)**2))
            ss_tot = float(np.sum((y - np.mean(y))**2))
            econ2_r2 = float(1 - ss_res/ss_tot) if ss_tot>0 else np.nan
        except Exception:
            pass

    summary = {
        "pooled": {
            "N_perception": int(len(P)),
            "econ1": {"N": int(mask1.sum()), "corr": econ1_corr, "R2": econ1_r2},
            "econ2": {"N": int(mask2.sum()), "corr": econ2_corr, "R2": econ2_r2},
        }
    }
    ensure_dir(out_root)
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Scatter for ECON2 (pooled)
    if mask2.sum() >= 5:
        x = P[mask2]
        y = Y2[mask2]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(x, y, s=10, alpha=0.3, color="#4C72B0")
        try:
            X = np.vstack([np.ones_like(x), x]).T
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            xs = np.linspace(min(x), max(x), 200)
            ys = beta[0] + beta[1]*xs
            ax.plot(xs, ys, color="#333", linestyle="--")
        except Exception:
            pass
        ax.set_xlabel("Perception (0-1)")
        ax.set_ylabel("ECON2 hours")
        ax.set_title("Heuristics: ECON2 vs Perception (pooled)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, "scatter_econ2_vs_perception_pooled.png"), dpi=200)
        plt.close(fig)

    # Bar correlations by state (pooled)
    out_state = []
    for st in ["NY", "TX"]:
        m1 = (~np.isnan(Y1)) & (ST==st)
        m2 = (~np.isnan(Y2)) & (ST==st)
        r1 = float(np.corrcoef(P[m1], Y1[m1])[0,1]) if (m1.sum()>=2 and np.std(P[m1])>0 and np.std(Y1[m1])>0) else np.nan
        r2 = float(np.corrcoef(P[m2], Y2[m2])[0,1]) if (m2.sum()>=2 and np.std(P[m2])>0 and np.std(Y2[m2])>0) else np.nan
        out_state.append({"STATE": st, "econ1_corr": r1, "econ2_corr": r2, "econ1_N": int(m1.sum()), "econ2_N": int(m2.sum())})
    with open(os.path.join(out_root, "pooled_corr_by_state.json"), "w", encoding="utf-8") as f:
        json.dump(out_state, f, indent=2)

    # Plot bars
    if out_state:
        labs = [d["STATE"] for d in out_state]
        v1 = [d["econ1_corr"] for d in out_state]
        v2 = [d["econ2_corr"] for d in out_state]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(labs, v1, color="#4C78A8")
        ax.set_ylabel("Pearson r")
        ax.set_title("Heuristics: Perception -> ECON1 corr (pooled by state)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, "bar_corr_econ1_by_state.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(labs, v2, color="#F58518")
        ax.set_ylabel("Pearson r")
        ax.set_title("Heuristics: Perception -> ECON2 corr (pooled by state)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, "bar_corr_econ2_by_state.png"), dpi=200)
        plt.close(fig)

    print(json.dumps(summary))


if __name__ == "__main__":
    main()


