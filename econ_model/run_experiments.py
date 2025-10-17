import os
import json
import numpy as np
import pandas as pd

from econ_model.data_prep import load_csv, build_features
from econ_model.utility import LinearUtility, ProspectUtility, RiskAverseLinear
from econ_model.decision import BinaryLogit, BinaryProbit, OLS, Huber
from econ_model.utils import to_json


def load_weeks(root: str):
    paths = [
        os.path.join(root, "survey", "survey_results", "week1", "Covid_W1_Full.csv"),
        os.path.join(root, "survey", "survey_results", "week2", "Covid_W2_Full.csv"),
        os.path.join(root, "survey", "survey_results", "week3", "Covid_W3_Full.csv"),
    ]
    dfs = [load_csv(p) for p in paths if os.path.exists(p)]
    if not dfs:
        dfs = [load_csv(os.path.join(root, "survey", "survey_results", "week1", "Covid_W1_NY.csv"))]
    return pd.concat(dfs, axis=0, ignore_index=True)


def split_mask(n: int, seed: int = 42, test_ratio: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_ratio))
    m = np.zeros(n, dtype=bool)
    m[idx[:cut]] = True
    return m


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(os.path.dirname(__file__), "results", "experiments")
    os.makedirs(results_dir, exist_ok=True)

    df = load_weeks(root)
    # two feature sets for future extension: minimal/full
    X_df, y_work_s, hours_s = build_features(df, feature_set="minimal")
    X = X_df.to_numpy(dtype=float)
    y = y_work_s.to_numpy(dtype=float)
    hours = hours_s.to_numpy(dtype=float)

    keep = ~np.isnan(y)
    Xc, yc = X[keep], y[keep]
    m = split_mask(len(Xc))
    Xtr, Xte, ytr, yte = Xc[m], Xc[~m], yc[m], yc[~m]

    # utility pool & decision pool
    utilities = {
        "linear_utility": LinearUtility(n_features=X.shape[1]),
        "prospect_utility": ProspectUtility(n_features=X.shape[1]),
        "risk_averse_linear": RiskAverseLinear(n_features=X.shape[1]),
    }
    decisions_cls = {
        "binary_logit": BinaryLogit,
        "binary_probit": BinaryProbit,
    }
    decisions_reg = {
        "ols": OLS,
        "huber": Huber,
    }

    # classification combos
    summary = {"classification": {}, "hours": {}}
    for u_name, util in utilities.items():
        for d_name, D in decisions_cls.items():
            model = D(util, l2=1e-2, max_iter=200)
            train_info = model.fit(Xtr, ytr)
            p_tr = model.predict_proba(Xtr)
            p_te = model.predict_proba(Xte)
            def metrics(y_true, p):
                eps = 1e-12
                yhat = (p>=0.5).astype(float)
                acc = float((yhat==y_true).mean())
                brier = float(np.mean((p - y_true)**2))
                logloss = float(-np.mean(y_true*np.log(p+eps)+(1-y_true)*np.log(1-p+eps)))
                return {"acc":acc,"brier":brier,"logloss":logloss}
            res = {
                "train": {**train_info, **metrics(ytr, p_tr)},
                "test": metrics(yte, p_te),
            }
            res["utility_params"] = util.get_params()
            summary["classification"][f"{u_name}+{d_name}"] = res
            # persist proba for plotting
            np.save(os.path.join(results_dir, f"{u_name}_{d_name}_work_proba_test.npy"), p_te)

    # hours combos (only y=1)
    keep_h = (y==1.0) & ~np.isnan(hours)
    Xh, yh = X[keep_h], hours[keep_h]
    mh = split_mask(len(Xh))
    Xh_tr, Xh_te, yh_tr, yh_te = Xh[mh], Xh[~mh], yh[mh], yh[~mh]
    for d_name, D in decisions_reg.items():
        reg = D()
        tr = reg.fit(Xh_tr, yh_tr)
        yp = reg.predict(Xh_te)
        rmse = float(np.sqrt(np.mean((yp - yh_te)**2)))
        mae = float(np.mean(np.abs(yp - yh_te)))
        res = {"train": tr, "test": {"rmse": rmse, "mae": mae}}
        res["coef_intercept"] = {"intercept": getattr(reg, "intercept_", None)}
        summary["hours"][d_name] = res
        # persist predictions
        np.save(os.path.join(results_dir, f"hours_{d_name}_pred_test.npy"), yp)
        np.save(os.path.join(results_dir, f"hours_y_test.npy"), yh_te)

    to_json(summary, os.path.join(results_dir, "summary.json"))
    # also persist y_test for classification
    np.save(os.path.join(results_dir, "work_y_test.npy"), yte)

    print("Experiments saved to:", results_dir)


if __name__ == "__main__":
    main()


