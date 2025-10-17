import os
import re
import json
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_meta_from_filename(filename: str) -> Tuple[str, int, str]:
    # Expect: {STATE}_W{week}_{SU_ID}.txt
    base = os.path.basename(filename)
    m = re.match(r"^([A-Z]{2})_W(\d)_([0-9]+)\.txt$", base)
    if not m:
        return "", -1, ""
    state, w, su_id = m.group(1), m.group(2), m.group(3)
    try:
        week = int(w)
    except Exception:
        week = -1
    return state, week, su_id


def collect_perception_outputs(out_root: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.isdir(out_root):
        return rows
    for wk in (1, 2, 3):
        wk_dir = os.path.join(out_root, f"week{wk}")
        if not os.path.isdir(wk_dir):
            continue
        for fn in os.listdir(wk_dir):
            if not fn.endswith(".txt"):
                continue
            path = os.path.join(wk_dir, fn)
            state, week, su_id = parse_meta_from_filename(path)
            if week < 0 or state == "" or su_id == "":
                continue
            content = read_text(path)
            try:
                value = float(content)
            except Exception:
                # Skip invalid
                continue
            if not (0.0 <= value <= 1.0):
                continue
            rows.append({
                "state": state,
                "week": week,
                "su_id": su_id,
                "prob": value,
                "path": path,
            })
    return rows


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def save_json(path: str, data) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def hist(ax, values: np.ndarray, bins: int = 30, label: str = None, rng=(0, 1)):
    ax.hist(values, bins=bins, range=rng, color="#4C78A8", alpha=0.7, label=label, edgecolor="white")
    ax.set_xlim(rng[0], rng[1])
    ax.set_xlabel("Probability (0-1)")
    ax.set_ylabel("Count")
    if label:
        ax.legend()


def overlay_hist(ax, values_a: np.ndarray, values_b: np.ndarray, label_a: str, label_b: str, bins: int = 30, rng=(0, 1)):
    ax.hist(values_a, bins=bins, range=rng, alpha=0.6, label=label_a, color="#4C78A8", edgecolor="white")
    ax.hist(values_b, bins=bins, range=rng, alpha=0.6, label=label_b, color="#F58518", edgecolor="white")
    ax.set_xlim(rng[0], rng[1])
    ax.set_xlabel("Probability (0-1)")
    ax.set_ylabel("Count")
    ax.legend()


def qq_uniform(ax, values: np.ndarray):
    # Simple QQ vs Uniform(0,1): sorted data vs theoretical quantiles
    n = len(values)
    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    x = np.sort(values)
    q = (np.arange(1, n + 1) - 0.5) / n  # uniform quantiles
    ax.scatter(q, x, s=8, color="#4C78A8")
    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Uniform quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_root = os.path.join(base_dir, "outputs", "perception")
    fig_root = os.path.join(base_dir, "outputs", "diagrams", "perception")
    ensure_dir(fig_root)

    rows = collect_perception_outputs(out_root)
    total = len(rows)

    # Aggregate arrays
    probs = np.array([r["prob"] for r in rows], dtype=float)
    weeks = np.array([r["week"] for r in rows], dtype=int)
    states = np.array([r["state"] for r in rows], dtype=object)

    # Save summary json
    summary = {
        "total_count": int(total),
        "count_by_week": {int(w): int(np.sum(weeks == w)) for w in (1, 2, 3)},
        "count_by_state": {s: int(np.sum(states == s)) for s in sorted(set(states))},
        "overall_mean": float(np.mean(probs)) if total > 0 else None,
        "overall_median": float(np.median(probs)) if total > 0 else None,
        "overall_std": float(np.std(probs)) if total > 0 else None,
        "p10": float(np.percentile(probs, 10)) if total > 0 else None,
        "p50": float(np.percentile(probs, 50)) if total > 0 else None,
        "p90": float(np.percentile(probs, 90)) if total > 0 else None,
    }
    save_json(os.path.join(fig_root, "perception_summary.json"), summary)

    # Overall histogram (fine bins)
    if total > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        hist(ax, probs, bins=100)
        ax.set_title("Perception probability - Overall")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_root, "perception_hist_overall.png"), dpi=200)
        plt.close(fig)

        # Zoomed near-zero ranges
        for rng, name in [((0, 0.10), "perception_hist_overall_zoom_0_0.10.png"), ((0, 0.05), "perception_hist_overall_zoom_0_0.05.png")]:
            fig, ax = plt.subplots(figsize=(7, 4))
            # more bins in a smaller window
            hist(ax, probs, bins=100, rng=rng)
            ax.set_title(f"Perception probability - Overall (zoom {rng[0]}-{rng[1]})")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_root, name), dpi=200)
            plt.close(fig)

    # Per-week subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for idx, w in enumerate((1, 2, 3)):
        vals = probs[weeks == w]
        hist(axes[idx], vals, bins=80, label=f"W{w}")
        axes[idx].set_title(f"Week {w}")
    fig.suptitle("Perception probability by week")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(fig_root, "perception_hist_by_week.png"), dpi=200)
    plt.close(fig)

    # Per-week zoomed near-zero subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for idx, w in enumerate((1, 2, 3)):
        vals = probs[weeks == w]
        hist(axes[idx], vals, bins=80, label=f"W{w}", rng=(0, 0.10))
        axes[idx].set_title(f"Week {w} (0-0.10)")
    fig.suptitle("Perception probability by week (zoom)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(fig_root, "perception_hist_by_week_zoom_0_0.10.png"), dpi=200)
    plt.close(fig)

    # Per-state overall overlay
    for s1, s2 in [("NY", "TX")]:
        if s1 in states and s2 in states:
            fig, ax = plt.subplots(figsize=(7, 4))
            overlay_hist(
                ax,
                probs[states == s1],
                probs[states == s2],
                label_a=s1,
                label_b=s2,
                bins=100,
            )
            ax.set_title("Perception probability - NY vs TX")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_root, "perception_hist_state_overlay.png"), dpi=200)
            plt.close(fig)

            # Zoomed state overlay near zero
            fig, ax = plt.subplots(figsize=(7, 4))
            overlay_hist(
                ax,
                probs[states == s1],
                probs[states == s2],
                label_a=s1,
                label_b=s2,
                bins=100,
                rng=(0, 0.10),
            )
            ax.set_title("Perception probability - NY vs TX (0-0.10)")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_root, "perception_hist_state_overlay_zoom_0_0.10.png"), dpi=200)
            plt.close(fig)

    # QQ vs Uniform overall
    if total > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        qq_uniform(ax, probs)
        ax.set_title("QQ plot vs Uniform(0,1) - Overall")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_root, "perception_qq_uniform_overall.png"), dpi=200)
        plt.close(fig)

    # Per week boxplot
    data_by_week = [probs[weeks == w] for w in (1, 2, 3)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data_by_week, labels=["W1", "W2", "W3"], vert=True, showfliers=False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability (0-1)")
    ax.set_title("Perception probability - Boxplot by week")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_root, "perception_box_by_week.png"), dpi=200)
    plt.close(fig)

    print(json.dumps(summary))


if __name__ == "__main__":
    main()


