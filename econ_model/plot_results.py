import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_calibration(y_true: np.ndarray, p: np.ndarray, out_path: str, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    frac_pos = []
    mean_pred = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        frac_pos.append(y_true[m].mean())
        mean_pred.append(p[m].mean())
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], 'k--', label='perfect')
    plt.plot(mean_pred, frac_pos, 'o-', label='model')
    plt.xlabel('Predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_pr(y_true: np.ndarray, p: np.ndarray, roc_path: str, pr_path: str):
    fpr, tpr, _ = roc_curve(y_true, p)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap = average_precision_score(y_true, p)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR')
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_dir: str):
    resid = y_true - y_pred
    plt.figure(figsize=(4, 4))
    plt.hist(resid, bins=30, alpha=0.8)
    plt.xlabel('Residual')
    plt.ylabel('Count')
    plt.title('Hours residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hours_residual_hist.png'))
    plt.close()
    plt.figure(figsize=(4, 4))
    plt.scatter(y_pred, y_true, s=6, alpha=0.5)
    plt.xlabel('Predicted hours')
    plt.ylabel('True hours')
    plt.title('Pred vs True (hours)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'hours_pred_vs_true.png'))
    plt.close()


def plot_cls_residual_hist(y_true: np.ndarray, p: np.ndarray, out_path: str, bins: int = 30, label: str = None):
    resid = y_true - p
    plt.figure(figsize=(4, 4))
    plt.hist(resid, bins=bins, alpha=0.85, label=label)
    plt.xlabel('Residual (y - p)')
    plt.ylabel('Count')
    plt.title('Classification residuals')
    if label:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_calibration_table(y_true: np.ndarray, p: np.ndarray, csv_path: str, n_bins: int = 10, strategy: str = 'uniform'):
    import csv
    y = y_true.astype(float)
    if strategy == 'quantile':
        qs = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(p, qs)
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    rows = [("bin_left", "bin_right", "n", "mean_pred", "frac_pos")]
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append((float(bins[b]), float(bins[b+1]), 0, None, None))
            continue
        mean_pred = float(p[m].mean())
        frac_pos = float(y[m].mean())
        rows.append((float(bins[b]), float(bins[b+1]), n, mean_pred, frac_pos))
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def plot_pointwise_scatter(y_true: np.ndarray, p: np.ndarray, out_path_scatter: str, out_path_resid: str):
    # y vs p 带抖动（pointwise），以及 |y-p| vs p 残差散点
    rng = np.random.default_rng(123)
    jitter = (rng.random(len(y_true)) - 0.5) * 0.1
    yj = y_true + jitter
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.scatter(p, yj, s=6, alpha=0.5)
    plt.xlabel('Predicted probability (p)')
    plt.ylabel('Observed y (jittered)')
    plt.title('y vs p (pointwise)')
    plt.tight_layout()
    plt.savefig(out_path_scatter)
    plt.close()
    plt.figure(figsize=(4,4))
    plt.scatter(p, np.abs(y_true - p), s=6, alpha=0.5)
    plt.xlabel('Predicted probability (p)')
    plt.ylabel('|y - p|')
    plt.title('Absolute residual vs p')
    plt.tight_layout()
    plt.savefig(out_path_resid)
    plt.close()


def main():
    root = os.path.dirname(__file__)
    results_dir = os.path.join(root, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    ensure_dir(plots_dir)

    with open(os.path.join(results_dir, 'baseline_summary.json'), 'r', encoding='utf-8') as f:
        summary = json.load(f)
    y_test = np.load(os.path.join(results_dir, 'work_y_test.npy'))
    p_test = np.load(os.path.join(results_dir, 'work_proba_test.npy'))
    yh_test = np.load(os.path.join(results_dir, 'hours_y_test.npy'))
    yp_test = np.load(os.path.join(results_dir, 'hours_pred_test.npy'))

    # 分类可视化（逐模型与对比）
    model_lin = 'linear_utility'
    model_pt = 'prospect_utility'
    decision = 'binary_logit'
    # 单模型命名
    plot_calibration(y_test, p_test, os.path.join(plots_dir, f'{model_lin}_{decision}_calibration.png'))
    plot_roc_pr(y_test, p_test,
                os.path.join(plots_dir, f'{model_lin}_{decision}_roc.png'),
                os.path.join(plots_dir, f'{model_lin}_{decision}_pr.png'))
    # 读取 PT 概率（若存在则绘制）
    p_test_pt_path = os.path.join(results_dir, 'work_proba_test_pt.npy')
    metrics = {}
    metrics['linear_brier'] = float(brier_score_loss(y_test, p_test))
    if os.path.exists(p_test_pt_path):
        p_test_pt = np.load(p_test_pt_path)
        plot_calibration(y_test, p_test_pt, os.path.join(plots_dir, f'{model_pt}_{decision}_calibration.png'))
        plot_roc_pr(y_test, p_test_pt,
                    os.path.join(plots_dir, f'{model_pt}_{decision}_roc.png'),
                    os.path.join(plots_dir, f'{model_pt}_{decision}_pr.png'))
        metrics['prospect_brier'] = float(brier_score_loss(y_test, p_test_pt))
        # 对比图（同图多曲线）
        # ROC
        fpr_lin, tpr_lin, _ = roc_curve(y_test, p_test)
        fpr_pt, tpr_pt, _ = roc_curve(y_test, p_test_pt)
        auc_lin = auc(fpr_lin, tpr_lin)
        auc_pt = auc(fpr_pt, tpr_pt)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr_lin, tpr_lin, label=f'{model_lin} AUC={auc_lin:.3f}')
        plt.plot(fpr_pt, tpr_pt, label=f'{model_pt} AUC={auc_pt:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_lin}_vs_{model_pt}_{decision}_roc.png'))
        plt.close()
        # PR
        prec_lin, rec_lin, _ = precision_recall_curve(y_test, p_test)
        prec_pt, rec_pt, _ = precision_recall_curve(y_test, p_test_pt)
        ap_lin = average_precision_score(y_test, p_test)
        ap_pt = average_precision_score(y_test, p_test_pt)
        plt.figure(figsize=(4, 4))
        plt.plot(rec_lin, prec_lin, label=f'{model_lin} AP={ap_lin:.3f}')
        plt.plot(rec_pt, prec_pt, label=f'{model_pt} AP={ap_pt:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_lin}_vs_{model_pt}_{decision}_pr.png'))
        plt.close()
        # Calibration compare（分箱曲线）
        def cal_points(y, p, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            idx = np.digitize(p, bins) - 1
            mp, fr = [], []
            for b in range(n_bins):
                m = idx == b
                if m.sum() == 0: continue
                fr.append(y[m].mean()); mp.append(p[m].mean())
            return np.array(mp), np.array(fr)
        mp_lin, fr_lin = cal_points(y_test, p_test)
        mp_pt, fr_pt = cal_points(y_test, p_test_pt)
        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(mp_lin, fr_lin, 'o-', label=model_lin)
        plt.plot(mp_pt, fr_pt, 's-', label=model_pt)
        plt.xlabel('Predicted probability'); plt.ylabel('Fraction of positives')
        plt.title('Calibration'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_lin}_vs_{model_pt}_{decision}_calibration.png'))
        plt.close()
        # 残差直方图对比（y - p）
        plt.figure(figsize=(4,4))
        plt.hist(y_test - p_test, bins=30, alpha=0.6, label=model_lin)
        plt.hist(y_test - p_test_pt, bins=30, alpha=0.6, label=model_pt)
        plt.xlabel('Residual (y - p)'); plt.ylabel('Count'); plt.title('Classification residuals')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_lin}_vs_{model_pt}_{decision}_residual_hist.png'))
        plt.close()
        # 保存两种分箱策略的校准表（uniform/quantile）
        save_calibration_table(y_test, p_test, os.path.join(plots_dir, f'{model_lin}_{decision}_calibration_uniform.csv'), n_bins=10, strategy='uniform')
        save_calibration_table(y_test, p_test, os.path.join(plots_dir, f'{model_lin}_{decision}_calibration_quantile.csv'), n_bins=10, strategy='quantile')
        save_calibration_table(y_test, p_test_pt, os.path.join(plots_dir, f'{model_pt}_{decision}_calibration_uniform.csv'), n_bins=10, strategy='uniform')
        save_calibration_table(y_test, p_test_pt, os.path.join(plots_dir, f'{model_pt}_{decision}_calibration_quantile.csv'), n_bins=10, strategy='quantile')
        # pointwise 散点
        plot_pointwise_scatter(y_test, p_test,
                               os.path.join(plots_dir, f'{model_lin}_{decision}_y_vs_p.png'),
                               os.path.join(plots_dir, f'{model_lin}_{decision}_abs_resid_vs_p.png'))
        plot_pointwise_scatter(y_test, p_test_pt,
                               os.path.join(plots_dir, f'{model_pt}_{decision}_y_vs_p.png'),
                               os.path.join(plots_dir, f'{model_pt}_{decision}_abs_resid_vs_p.png'))

    # 小时可视化（基线 OLS；命名按约定）
    plot_residuals(yh_test, yp_test, plots_dir)
    # 复制命名：linear_utility_ols_*
    os.replace(os.path.join(plots_dir, 'hours_residual_hist.png'),
               os.path.join(plots_dir, 'linear_utility_ols_hours_residual_hist.png'))
    os.replace(os.path.join(plots_dir, 'hours_pred_vs_true.png'),
               os.path.join(plots_dir, 'linear_utility_ols_hours_pred_vs_true.png'))

    with open(os.path.join(results_dir, 'plots', 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('Plots saved to:', plots_dir)


if __name__ == '__main__':
    main()


