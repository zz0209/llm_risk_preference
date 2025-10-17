import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def ensure(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    root = os.path.dirname(__file__)
    exp_dir = os.path.join(root, 'results', 'experiments')
    plots = os.path.join(exp_dir, 'plots')
    ensure(plots)

    with open(os.path.join(exp_dir, 'summary.json'), 'r', encoding='utf-8') as f:
        summary = json.load(f)
    y_test = np.load(os.path.join(exp_dir, 'work_y_test.npy'))

    # classification comparisons (ROC/PR/calibration-like by quantile points)
    combos = list(summary['classification'].keys())
    # 先读取全部组合的 p，确保索引对齐（run_experiments 用同一划分，长度应一致）
    p_map = {}
    for k in combos:
        path = os.path.join(exp_dir, f"{k.replace('+','_')}_work_proba_test.npy")
        if os.path.exists(path):
            p_map[k] = np.load(path)
    # 取共同长度
    min_len = min(len(arr) for arr in p_map.values()) if p_map else 0
    y_use = y_test[:min_len]
    # 统一裁剪与掩码（对所有组合使用同一子集，防止错位）
    valid = np.isfinite(y_use)
    for k, arr in list(p_map.items()):
        p = np.clip(arr[:min_len], 1e-6, 1-1e-6)
        valid = valid & np.isfinite(p)
        p_map[k] = p
    # 应用共同掩码
    y_use = y_use[valid]
    probas = {k: v[valid] for k, v in p_map.items()}

    # ROC（叠加对比）
    plt.figure(figsize=(5,5))
    for k, p in probas.items():
        fpr, tpr, _ = roc_curve(y_use, p)
        plt.plot(fpr, tpr, label=f"{k} AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (all combos)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots, 'all_combos_binary_logit_roc.png'))
    plt.close()

    # PR（叠加对比）
    plt.figure(figsize=(5,5))
    for k, p in probas.items():
        prec, rec, _ = precision_recall_curve(y_use, p)
        ap = average_precision_score(y_use, p)
        plt.plot(rec, prec, label=f"{k} AP={ap:.3f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR (all combos)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots, 'all_combos_binary_logit_pr.png'))
    plt.close()

    # Calibration（quantile 分箱叠加对比）
    def cal_points(y, p, n_bins=10):
        qs = np.linspace(0,1,n_bins+1)
        p = np.clip(p, 1e-6, 1-1e-6)
        bins = np.quantile(p, qs)
        bins[0], bins[-1] = 0.0, 1.0
        idx = np.digitize(p, bins) - 1
        mp, fr = [], []
        for b in range(n_bins):
            m = idx==b
            if m.sum()==0: continue
            mp.append(p[m].mean()); fr.append(y_use[m].mean())
        return np.array(mp), np.array(fr)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'k--',label='perfect')
    for k, p in probas.items():
        mp, fr = cal_points(y_test, p)
        plt.plot(mp, fr, 'o-', label=k)
    plt.xlabel('Predicted probability'); plt.ylabel('Fraction of positives')
    plt.title('Calibration (all combos, quantile-binned)')
    plt.xlim(0,1); plt.ylim(0,1)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots, 'all_combos_binary_logit_calibration.png'))
    plt.close()

    # Classification residual hist（叠加对比，y - p）
    plt.figure(figsize=(5,4))
    for k, p in probas.items():
        plt.hist(y_use - p, bins=30, alpha=0.5, label=k)
    plt.xlabel('Residual (y - p)'); plt.ylabel('Count')
    plt.title('Classification residuals (all combos)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plots, 'all_combos_binary_logit_residual_hist.png'))
    plt.close()

    # 另外分别导出 Probit-only 与 Logit-only 的对比，避免命名误导
    def filter_combos(keyword):
        return {k: v for k, v in probas.items() if keyword in k}
    for name, subset in [("binary_logit", filter_combos("binary_logit")), ("binary_probit", filter_combos("binary_probit"))]:
        if not subset: continue
        # ROC
        plt.figure(figsize=(5,5))
        for k, p in subset.items():
            fpr, tpr, _ = roc_curve(y_use, p)
            plt.plot(fpr, tpr, label=f"{k} AUC={auc(fpr,tpr):.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC ({name} only)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots, f'all_combos_{name}_only_roc.png'))
        plt.close()
        # PR
        plt.figure(figsize=(5,5))
        for k, p in subset.items():
            prec, rec, _ = precision_recall_curve(y_use, p)
            ap = average_precision_score(y_use, p)
            plt.plot(rec, prec, label=f"{k} AP={ap:.3f}")
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR ({name} only)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots, f'all_combos_{name}_only_pr.png'))
        plt.close()
        # Calibration
        plt.figure(figsize=(5,5))
        plt.plot([0,1],[0,1],'k--',label='perfect')
        for k, p in subset.items():
            mp, fr = cal_points(y_use, p)
            plt.plot(mp, fr, 'o-', label=k)
        plt.xlabel('Predicted probability'); plt.ylabel('Fraction of positives')
        plt.title(f'Calibration ({name} only)')
        plt.xlim(0,1); plt.ylim(0,1)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots, f'all_combos_{name}_only_calibration.png'))
        plt.close()

    # hours comparisons
    yh = np.load(os.path.join(exp_dir, 'hours_y_test.npy'))
    # 单模型各自图 + 叠加对比图
    preds = {}
    for d_name in ['ols', 'huber']:
        pred_path = os.path.join(exp_dir, f"hours_{d_name}_pred_test.npy")
        if os.path.exists(pred_path):
            preds[d_name] = np.load(pred_path)
            # 单模型残差直方图
            plt.figure(figsize=(5,4))
            plt.hist(yh - preds[d_name], bins=30, alpha=0.8)
            plt.xlabel('Residual (hours)'); plt.ylabel('Count')
            plt.title(f'Hours residuals - {d_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots, f'all_combos_{d_name}_hours_residual_hist.png'))
            plt.close()
            # 单模型 Pred vs True
            plt.figure(figsize=(5,4))
            plt.scatter(preds[d_name], yh, s=6, alpha=0.5)
            plt.xlabel('Predicted hours'); plt.ylabel('True hours')
            plt.title(f'Pred vs True (hours) - {d_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots, f'all_combos_{d_name}_hours_pred_vs_true.png'))
            plt.close()
    if len(preds) >= 2:
        # 叠加残差直方图
        plt.figure(figsize=(5,4))
        for d_name, yp in preds.items():
            plt.hist(yh - yp, bins=30, alpha=0.5, label=d_name)
        plt.xlabel('Residual (hours)'); plt.ylabel('Count')
        plt.title('Hours residuals (all regressors)')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots, 'all_combos_hours_residual_hist.png'))
        plt.close()
        # 叠加 Pred vs True
        plt.figure(figsize=(5,4))
        for d_name, yp in preds.items():
            plt.scatter(yp, yh, s=6, alpha=0.4, label=d_name)
        plt.xlabel('Predicted hours'); plt.ylabel('True hours')
        plt.title('Pred vs True (hours) - all regressors')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots, 'all_combos_hours_pred_vs_true.png'))
        plt.close()

    print('Comparison plots saved to:', plots)


if __name__ == '__main__':
    main()


