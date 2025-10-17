import os
import re
import json
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_heuristic_outputs() -> pd.DataFrame:
    root = os.path.join(PROJECT_ROOT, "outputs", "ECON_perception_heuritics")
    rows: List[Dict[str, str]] = []
    pat = re.compile(r"^([A-Z]{2})_W(\d)_([0-9]+)_(ECON1|ECON2|PERCEPTION)\.txt$")
    for w in (1, 2, 3):
        d = os.path.join(root, f"week{w}")
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".txt"):
                continue
            m = pat.match(fn)
            if not m:
                continue
            state, ww, su, kind = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            content = read_text(os.path.join(d, fn))
            rows.append({"STATE": state, "WEEK": ww, "SU_ID": su, "KIND": kind, "content": content})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # pivot to wide per (state, week, su)
    wide = df.pivot_table(index=["STATE", "WEEK", "SU_ID"], columns="KIND", values="content", aggfunc="first").reset_index()
    # parse
    def econ1_worked(s: str) -> float:
        if not isinstance(s, str):
            return np.nan
        m = re.search(r"\((\d+)\)", s)
        if not m:
            return np.nan
        code = m.group(1)
        return 1.0 if code in {"1", "2"} else 0.0 if code == "3" else np.nan
    def econ2_hours(s: str) -> float:
        if not isinstance(s, str):
            return np.nan
        s2 = s.strip()
        if re.match(r"^\d+$", s2):
            v = int(s2)
            return float(v) if 0 <= v <= 168 else np.nan
        m = re.match(r"\s*(\d+)", s2)
        if m:
            v = int(m.group(1))
            return float(v) if 0 <= v <= 168 else np.nan
        return np.nan
    def perception_float(s: str) -> float:
        try:
            v = float(s)
            return v if 0.0 <= v <= 1.0 else np.nan
        except Exception:
            return np.nan
    wide["work"] = wide.get("ECON1").apply(econ1_worked) if "ECON1" in wide.columns else np.nan
    wide["hours"] = wide.get("ECON2").apply(econ2_hours) if "ECON2" in wide.columns else np.nan
    wide["p_hat"] = wide.get("PERCEPTION").apply(perception_float) if "PERCEPTION" in wide.columns else np.nan
    return wide


def load_env_windows() -> Dict[Tuple[str, int], Dict[str, float]]:
    # compute avg new cases / deaths per state-week
    st_csv = os.path.join(PROJECT_ROOT, "data", "ny_tx_2020-04-10_to_2020-06-20_slim.csv")
    st = pd.read_csv(st_csv)
    st["Report_Date"] = pd.to_datetime(st["Report_Date"]) if "Report_Date" in st.columns else pd.to_datetime(st.iloc[:,0])
    windows = {
        1: (pd.Timestamp("2020-04-12"), pd.Timestamp("2020-04-20")),
        2: (pd.Timestamp("2020-04-26"), pd.Timestamp("2020-05-04")),
        3: (pd.Timestamp("2020-05-24"), pd.Timestamp("2020-05-30")),
    }
    state_name = {"NY": "New York", "TX": "Texas"}
    feats: List[Dict[str, float]] = []
    for w, (start, end) in windows.items():
        for st_code, st_full in state_name.items():
            sub = st[(st["Province_State"]==st_full) & (st["Report_Date"]>=start) & (st["Report_Date"]<=end)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("Report_Date")
            # compute diffs
            new_cases = sub["Confirmed"].diff().dropna() if "Confirmed" in sub.columns else pd.Series(dtype=float)
            new_deaths = sub["Deaths"].diff().dropna() if "Deaths" in sub.columns else pd.Series(dtype=float)
            feats.append({
                "STATE": st_code,
                "WEEK": w,
                "avg_new_cases": float(new_cases.mean()) if len(new_cases)>0 else np.nan,
                "avg_new_deaths": float(new_deaths.mean()) if len(new_deaths)>0 else np.nan,
            })
    env = pd.DataFrame(feats)
    # normalize to [0,1]
    def norm01(s: pd.Series) -> pd.Series:
        s2 = s.astype(float)
        lo, hi = float(np.nanmin(s2)), float(np.nanmax(s2))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(np.full_like(s2, np.nan))
        return (s2 - lo) / (hi - lo)
    env["nc_norm"] = norm01(env["avg_new_cases"]) if "avg_new_cases" in env.columns else np.nan
    env["nd_norm"] = norm01(env["avg_new_deaths"]) if "avg_new_deaths" in env.columns else np.nan
    env["p_obj_raw"] = 0.7 * env["nc_norm"].astype(float) + 0.3 * env["nd_norm"].astype(float)
    env["p_obj"] = env["p_obj_raw"].clip(1e-6, 1-1e-6)
    # map to dict
    m: Dict[Tuple[str, int], float] = {}
    for _, r in env.iterrows():
        m[(r["STATE"], int(r["WEEK"]))] = float(r["p_obj"]) if np.isfinite(r["p_obj"]) else np.nan
    return m


def prelec(p: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.exp(-beta * ((-np.log(p)) ** alpha))


def fit_prelec(p_obj: np.ndarray, p_hat: np.ndarray) -> Tuple[float, float, float]:
    # grid search for stability
    alphas = np.linspace(0.5, 1.5, 21)
    betas = np.linspace(0.5, 2.0, 31)
    best = (1.0, 1.0, 1e9)
    for a in alphas:
        pw = (-np.log(np.clip(p_obj, 1e-9, 1-1e-9))) ** a
        for b in betas:
            pred = np.exp(-b * pw)
            mse = float(np.mean((pred - p_hat) ** 2))
            if mse < best[2]:
                best = (float(a), float(b), mse)
    return best


def ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    X = np.vstack([np.ones_like(x), x]).T
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "diagrams", "perception_heuritics_explainability", "eut_pt")
    ensure_dir(out_dir)

    df = load_heuristic_outputs()
    env_map = load_env_windows()
    if df.empty:
        print("{}")
        return
    # attach p_obj
    df["p_obj"] = df.apply(lambda r: env_map.get((r["STATE"], int(r["WEEK"]))), axis=1)
    # drop missing
    d = df.dropna(subset=["p_obj", "p_hat"]).copy()
    p_obj = d["p_obj"].astype(float).values
    p_hat = d["p_hat"].astype(float).values

    # fit Prelec p_hat ≈ w(p_obj)
    a, b, mse = fit_prelec(p_obj, p_hat)
    w_obj = prelec(p_obj, a, b)

    # EUT vs PT explainability for ECON1/ECON2 (pooled)
    res = {"prelec": {"alpha": a, "beta": b, "mse": mse}}
    # ECON1
    y1 = d["work"].astype(float).values
    m1 = ~np.isnan(y1)
    r2_eut_econ1 = ols_r2(y1[m1], p_hat[m1])
    r2_pt_econ1 = ols_r2(y1[m1], w_obj[m1])
    res["econ1"] = {"N": int(np.sum(m1)), "R2_eut": r2_eut_econ1, "R2_pt": r2_pt_econ1}
    # ECON2
    y2 = d["hours"].astype(float).values
    m2 = ~np.isnan(y2)
    r2_eut_econ2 = ols_r2(y2[m2], p_hat[m2])
    r2_pt_econ2 = ols_r2(y2[m2], w_obj[m2])
    res["econ2"] = {"N": int(np.sum(m2)), "R2_eut": r2_eut_econ2, "R2_pt": r2_pt_econ2}

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # plots
    # calibration: p_hat vs p_obj with fitted prelec curve
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(p_obj, p_hat, s=10, alpha=0.35, color="#4C72B0", label="LLM")
    xs = np.linspace(1e-3, 0.999, 200)
    ax.plot(xs, prelec(xs, a, b), color="#C44E52", label=f"Prelec fit (α={a:.2f}, β={b:.2f})")
    ax.plot([0,1],[0,1], linestyle="--", color="#888")
    ax.set_xlabel("Objective risk p_obj")
    ax.set_ylabel("LLM perceived p_hat")
    ax.set_title("Calibration: p_hat vs p_obj")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_prelec.png"), dpi=200)
    plt.close(fig)

    # R2 comparison bars
    fig, ax = plt.subplots(figsize=(6,4))
    cats = ["ECON1", "ECON2"]
    eut = [r2_eut_econ1, r2_eut_econ2]
    pt  = [r2_pt_econ1, r2_pt_econ2]
    x = np.arange(len(cats))
    w = 0.38
    ax.bar(x - w/2, eut, w, label="EUT (p_hat)", color="#4C78A8")
    ax.bar(x + w/2, pt,  w, label="PT (w(p_obj))", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("OLS R²")
    ax.set_title("EUT vs PT (pooled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "r2_eut_vs_pt.png"), dpi=200)
    plt.close(fig)

    # dump simple print for console
    print(json.dumps(res))


if __name__ == "__main__":
    main()


