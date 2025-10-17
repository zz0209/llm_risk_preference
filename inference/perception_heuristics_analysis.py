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
    pat = re.compile(r"^([A-Z]{2})_W(\d)_([0-9]+)_(ECON1|ECON2|PERCEPTION)\.txt$")
    if not os.path.isdir(root):
        return rows
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


def hist(ax, values: np.ndarray, bins: int = 100, rng=(0, 1)):
    ax.hist(values, bins=bins, range=rng, color="#4C78A8", edgecolor="white", alpha=0.8)
    ax.set_xlim(rng)
    ax.set_xlabel("Probability (0-1)")
    ax.set_ylabel("Count")


def qq_uniform(ax, values: np.ndarray):
    n = len(values)
    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    x = np.sort(values)
    q = (np.arange(1, n + 1) - 0.5) / n
    ax.scatter(q, x, s=10, alpha=0.6, color="#4C78A8")
    ax.plot([0, 1], [0, 1], color="#333", linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Uniform quantiles")
    ax.set_ylabel("Empirical quantiles")


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_root = os.path.join(base, "outputs", "ECON_perception_heuritics")
    fig_root = os.path.join(base, "outputs", "diagrams", "perception_heuritics")
    ensure_dir(fig_root)

    rows = iter_outputs(in_root)
    # Filter perception only
    per_rows = [r for r in rows if r["KIND"] == "PERCEPTION"]
    vals: List[float] = []
    states: List[str] = []
    for r in per_rows:
        try:
            v = float(r["content"])
        except Exception:
            continue
        if 0.0 <= v <= 1.0:
            vals.append(v)
            states.append(r["STATE"])
    vals = np.array(vals, dtype=float)
    states = np.array(states, dtype=object)

    summary = {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)) if len(vals)>0 else None,
        "median": float(np.median(vals)) if len(vals)>0 else None,
        "std": float(np.std(vals)) if len(vals)>0 else None,
        "p10": float(np.percentile(vals, 10)) if len(vals)>0 else None,
        "p90": float(np.percentile(vals, 90)) if len(vals)>0 else None,
        "NY_count": int(np.sum(states=="NY")),
        "TX_count": int(np.sum(states=="TX")),
    }
    with open(os.path.join(fig_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # overall histogram
    if len(vals) > 0:
        fig, ax = plt.subplots(figsize=(7,4))
        hist(ax, vals, bins=100, rng=(0,1))
        ax.set_title("Heuristics perception - overall")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_root, "hist_overall.png"), dpi=200)
        plt.close(fig)

        # zooms
        for rng, name in [((0,0.10), "hist_overall_zoom_0_0.10.png"), ((0,0.05), "hist_overall_zoom_0_0.05.png")]:
            fig, ax = plt.subplots(figsize=(7,4))
            hist(ax, vals, bins=100, rng=rng)
            ax.set_title(f"Heuristics perception - overall (zoom {rng[0]}-{rng[1]})")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_root, name), dpi=200)
            plt.close(fig)

        # state overlay
        for s1, s2 in [("NY","TX")]:
            fig, ax = plt.subplots(figsize=(7,4))
            a = vals[states==s1]
            b = vals[states==s2]
            ax.hist(a, bins=100, range=(0,1), alpha=0.6, label=s1, color="#4C78A8", edgecolor="white")
            ax.hist(b, bins=100, range=(0,1), alpha=0.6, label=s2, color="#F58518", edgecolor="white")
            ax.set_xlim(0,1)
            ax.set_xlabel("Probability (0-1)")
            ax.set_ylabel("Count")
            ax.legend()
            ax.set_title("Heuristics perception - NY vs TX")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_root, "hist_state_overlay.png"), dpi=200)
            plt.close(fig)

        # QQ vs uniform
        fig, ax = plt.subplots(figsize=(5,5))
        qq_uniform(ax, vals)
        ax.set_title("Heuristics perception - QQ vs Uniform")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_root, "qq_uniform_overall.png"), dpi=200)
        plt.close(fig)

    print(json.dumps(summary))


if __name__ == "__main__":
    main()


