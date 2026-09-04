import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from t13 import config as C
from t13.data import load_synb2b, build_synb2b_features, SYNB2B_FEATURES
from t13.drift import plan_events, apply_covariate_shift

raw = load_synb2b()
n = len(raw); n_train = int(round(C.TRAIN_FRACTION * n))
days = raw["day"].to_numpy(float)
ev_cov = plan_events(days, n_train)[0]
shifted = apply_covariate_shift(raw, ev_cov)
feats = build_synb2b_features(shifted)
ev = slice(n_train, n)

for name, v in [
    ("supplier_age<30", (feats["supplier_age_days"].to_numpy() < 30).astype(float)),
    ("supplier_age<7", (feats["supplier_age_days"].to_numpy() < 7).astype(float)),
    ("edge_n<3", (feats["edge_n"].to_numpy() < 3).astype(float)),
    ("edge_n<1", (feats["edge_n"].to_numpy() < 1).astype(float)),
    ("edge_n<6", (feats["edge_n"].to_numpy() < 6).astype(float)),
    ("sup_n<10", (feats["sup_n"].to_numpy() < 10).astype(float)),
]:
    tr, te = v[:n_train], v[n_train:]
    q1 = te[: len(te) // 3].mean(); q3 = te[-len(te) // 3:].mean()
    print(f"{name:18s} train {tr.mean():.4f}  eval {te.mean():.4f}  "
          f"eval-first-third {q1:.4f} eval-last-third {q3:.4f}")

# scorer importances on a plain fit against the base labels
import lightgbm as lgb
X = feats[SYNB2B_FEATURES].to_numpy(float)
y = shifted["base_label"].to_numpy(int)
clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=1, verbose=-1, random_state=11)
clf.fit(X[:n_train], y[:n_train])
imp = sorted(zip(SYNB2B_FEATURES, clf.feature_importances_), key=lambda t: -t[1])
print("\ntop scorer features:")
for k, v in imp[:10]:
    print(f"  {k:22s} {v}")
from sklearn.metrics import average_precision_score, roc_auc_score
s = clf.predict_proba(X)[:, 1]
print("base-label AUC-PR eval", average_precision_score(y[n_train:], s[n_train:]),
      "AUC-ROC", roc_auc_score(y[n_train:], s[n_train:]))
print("eval base pos rate", y[n_train:].mean())
