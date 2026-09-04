"""The frozen scorer.

One LightGBM fit on the first 20% of the stream in time order, then no refit for the
remaining 80%. The point is not that freezing a model is good practice - it is not -
but that freezing it isolates the calibration layer. If the scorer were refitted the
question "did the decision layer hold risk" would be entangled with "did the model
recover", and neither answer would be readable.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.0575,
    num_leaves=31,
    min_child_samples=25,
    subsample=1.0,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


def train_frozen_scorer(X, y, n_train: int, seed: int):
    import lightgbm as lgb

    Xtr = np.ascontiguousarray(X[:n_train], dtype=np.float64)
    ytr = np.asarray(y[:n_train], dtype=int)
    if ytr.sum() < 5:
        raise RuntimeError(f"training split carries only {int(ytr.sum())} positives")
    clf = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
    clf.fit(Xtr, ytr)
    scores = clf.predict_proba(np.ascontiguousarray(X, dtype=np.float64))[:, 1]
    return clf, scores


def scorer_report(scores: np.ndarray, y: np.ndarray, n_train: int) -> dict:
    ev_s, ev_y = scores[n_train:], np.asarray(y[n_train:], dtype=int)
    return {
        "auc_pr_eval": float(average_precision_score(ev_y, ev_s)),
        "auc_roc_eval": float(roc_auc_score(ev_y, ev_s)),
        "n_eval": int(len(ev_y)),
        "n_pos_eval": int(ev_y.sum()),
        "pos_rate_eval": float(ev_y.mean()),
    }


def post_event_degradation(scores, y, days, events, horizon_days: float = 10.0) -> dict:
    """AUC-PR in the ten days after each drift event, against the ten days before."""
    out = {}
    y = np.asarray(y, dtype=int)
    for ev in events:
        pre = (days >= ev.day - horizon_days) & (days < ev.day)
        post = (days >= ev.day) & (days < ev.day + horizon_days)
        rec = {}
        for name, m in (("pre", pre), ("post", post)):
            yy, ss = y[m], scores[m]
            rec[f"n_{name}"] = int(m.sum())
            rec[f"pos_{name}"] = int(yy.sum())
            rec[f"auc_pr_{name}"] = (
                float(average_precision_score(yy, ss)) if 0 < yy.sum() < len(yy) else None
            )
        if rec["auc_pr_pre"] and rec["auc_pr_post"] is not None:
            rec["delta_auc_pr"] = rec["auc_pr_post"] - rec["auc_pr_pre"]
            rec["rel_drop"] = 1.0 - rec["auc_pr_post"] / rec["auc_pr_pre"]
        out[ev.kind] = rec
    return out
