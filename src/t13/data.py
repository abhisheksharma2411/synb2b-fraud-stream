"""Stream loading and strictly causal feature construction.

Two streams are supported. SynB2B-Fraud is the primary one; the ULB card portfolio
is the external-validity check. Both are delivered in time order and never shuffled.

Every feature here is computed in a single forward pass over the stream, so the
value attached to row i depends only on rows 0..i-1. That is checked in
tests/test_causality.py rather than asserted in prose.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd

from .config import SECONDS_PER_DAY

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

SYNB2B_CSV = os.path.join(RAW_DIR, "synb2b_fraud.csv")
ULB_PARQUET = os.path.join(RAW_DIR, "ulb_creditcard.parquet")

SYNB2B_ZENODO = "https://zenodo.org/records/21668281/files/{name}?download=1"


def ensure_synb2b() -> str:
    if not os.path.exists(SYNB2B_CSV):
        import urllib.request

        os.makedirs(RAW_DIR, exist_ok=True)
        urllib.request.urlretrieve(SYNB2B_ZENODO.format(name="synb2b_fraud.csv"), SYNB2B_CSV)
    return SYNB2B_CSV


def ensure_ulb() -> str:
    if not os.path.exists(ULB_PARQUET):
        from sklearn.datasets import fetch_openml

        os.makedirs(RAW_DIR, exist_ok=True)
        frame = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
        frame.to_parquet(ULB_PARQUET, index=False)
    return ULB_PARQUET


def load_synb2b(small: bool = False) -> pd.DataFrame:
    df = pd.read_csv(ensure_synb2b())
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if small:
        # Every fourth row rather than a head slice. Truncating the head would squeeze
        # eighteen months of calendar into four and land the drift windows on top of
        # each other; a stride keeps the span, the base rate and the event placement.
        df = df.iloc[::4].reset_index(drop=True)
    df["day"] = (df["timestamp"] - df["timestamp"].iloc[0]) / SECONDS_PER_DAY
    df["base_label"] = df["fraud_injected"].astype(np.int8)
    df["topology"] = df["fraud_topology"].fillna("")
    return df


def load_ulb(small: bool = False) -> pd.DataFrame:
    """OpenML data_id=1597 ships V1..V28, Amount and Class - no `Time` column.

    The rows arrive in the source file's original temporal order, so the row index
    is the time index. Nothing is sorted and nothing is shuffled. See DECISIONS.md
    D-03 for why that is the only honest reading of this distribution of the data.
    """
    df = pd.read_parquet(ensure_ulb()).reset_index(drop=True)
    if small:
        df = df.iloc[::5].reset_index(drop=True)
    n = len(df)
    # The published portfolio covers 48 hours of card traffic; 48h of decisions is far
    # too short for a 180-day reconciliation window to mean anything, so the stream is
    # replayed on the SynB2B calendar: the same ordering, stretched over 561.34 days.
    df["day"] = np.linspace(0.0, 561.34, n)
    df["timestamp"] = (df["day"] * SECONDS_PER_DAY).astype(np.int64)
    df["base_label"] = df["Class"].astype(int).astype(np.int8)
    df["amount"] = df["Amount"].astype(float)
    df["topology"] = np.where(df["base_label"].to_numpy() == 1, "ULB_card", "")
    return df


# ---------------------------------------------------------------------------
# Causal features
# ---------------------------------------------------------------------------
SYNB2B_FEATURES = [
    "log_amount", "terms", "supplier_age_days", "supplier_novel",
    "edge_n", "edge_days_since", "edge_amt_z", "edge_term_z",
    "bank_changed", "sup_n_buyers", "sup_n", "buyer_n",
    "amt_over_edge_max", "bank_n_suppliers", "cat_code", "country_code",
    "sup_age_lt7", "edge_amt_ratio",
]


def build_synb2b_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single forward pass. Row i sees only rows < i."""
    n = len(df)
    ts = df["timestamp"].to_numpy(np.int64)
    amt = df["amount"].to_numpy(float)
    terms = df["payment_terms_days"].to_numpy(float)
    buyer = df["buyer_id"].to_numpy(object)
    supp = df["supplier_id"].to_numpy(object)
    bank = df["supplier_bank_account"].to_numpy(object)
    cat = pd.Categorical(df["supplier_category"]).codes.astype(float)
    ctry = pd.Categorical(df["buyer_country"]).codes.astype(float)

    out = {k: np.zeros(n, dtype=float) for k in SYNB2B_FEATURES}
    log_amt = np.log1p(amt)

    sup_first = {}
    sup_count = defaultdict(int)
    sup_buyers = defaultdict(set)
    buyer_count = defaultdict(int)
    bank_suppliers = defaultdict(set)
    # running edge state: [count, sum_log_amt, sumsq_log_amt, sum_term, sumsq_term,
    #                      last_ts, last_bank, max_amt]
    edge_n = defaultdict(int)
    edge_s1 = defaultdict(float)
    edge_s2 = defaultdict(float)
    edge_t1 = defaultdict(float)
    edge_t2 = defaultdict(float)
    edge_last = {}
    edge_bank = {}
    edge_max = defaultdict(float)
    edge_mean_amt = defaultdict(float)

    for i in range(n):
        b, s, ba = buyer[i], supp[i], bank[i]
        e = (b, s)
        if s not in sup_first:
            sup_first[s] = ts[i]
        age = (ts[i] - sup_first[s]) / SECONDS_PER_DAY
        out["log_amount"][i] = log_amt[i]
        out["terms"][i] = terms[i]
        out["supplier_age_days"][i] = age
        out["supplier_novel"][i] = 1.0 if age < 30.0 else 0.0
        out["sup_age_lt7"][i] = 1.0 if age < 7.0 else 0.0
        out["sup_n"][i] = sup_count[s]
        out["sup_n_buyers"][i] = len(sup_buyers[s])
        out["buyer_n"][i] = buyer_count[b]
        out["bank_n_suppliers"][i] = len(bank_suppliers[ba])
        out["cat_code"][i] = cat[i]
        out["country_code"][i] = ctry[i]

        k = edge_n[e]
        out["edge_n"][i] = k
        out["edge_days_since"][i] = (
            (ts[i] - edge_last[e]) / SECONDS_PER_DAY if e in edge_last else -1.0
        )
        if k >= 2:
            m = edge_s1[e] / k
            v = max(edge_s2[e] / k - m * m, 1e-9)
            out["edge_amt_z"][i] = (log_amt[i] - m) / np.sqrt(v)
            mt = edge_t1[e] / k
            vt = max(edge_t2[e] / k - mt * mt, 1e-9)
            out["edge_term_z"][i] = (terms[i] - mt) / np.sqrt(vt)
        out["amt_over_edge_max"][i] = amt[i] / edge_max[e] if edge_max[e] > 0 else 0.0
        out["edge_amt_ratio"][i] = (
            amt[i] / edge_mean_amt[e] if edge_mean_amt[e] > 0 else 0.0
        )
        out["bank_changed"][i] = (
            1.0 if (e in edge_bank and edge_bank[e] != ba) else 0.0
        )

        sup_count[s] += 1
        sup_buyers[s].add(b)
        buyer_count[b] += 1
        bank_suppliers[ba].add(s)
        edge_n[e] = k + 1
        edge_s1[e] += log_amt[i]
        edge_s2[e] += log_amt[i] * log_amt[i]
        edge_t1[e] += terms[i]
        edge_t2[e] += terms[i] * terms[i]
        edge_last[e] = ts[i]
        edge_bank[e] = ba
        edge_max[e] = max(edge_max[e], amt[i])
        edge_mean_amt[e] = (edge_mean_amt[e] * k + amt[i]) / (k + 1)

    return pd.DataFrame(out, index=df.index)


ULB_FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount"]


def build_ulb_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[f"V{i}" for i in range(1, 29)]].astype(float).copy()
    out["log_amount"] = np.log1p(df["amount"].to_numpy(float))
    return out


def load_stream(stream: str, small: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    if stream == "synb2b":
        raw = load_synb2b(small=small)
        return raw, build_synb2b_features(raw), SYNB2B_FEATURES
    if stream == "ulb":
        raw = load_ulb(small=small)
        return raw, build_ulb_features(raw), ULB_FEATURES
    raise ValueError(f"unknown stream {stream!r}")
