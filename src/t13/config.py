"""Configuration objects and the stipulated operating constants for T13.

Every constant that is a modelling *choice* rather than a measurement lives here so
that DECISIONS.md can point at exactly one file.
"""
from __future__ import annotations

import json as _json
import os as _os
from dataclasses import dataclass, field, asdict
from typing import Dict

SECONDS_PER_DAY = 86400.0

# ----------------------------------------------------------------------------
# Cost constants (stipulated; see DECISIONS.md D-08 and the sensitivity ablation)
# ----------------------------------------------------------------------------
# Fraud allowed through: the unrecovered share of the invoice plus fixed handling
# of the reconciliation exception / representment file.
C_FN_RATE = 0.72          # share of invoice value not recovered after the fact
C_FN_FIXED = 180.0        # USD, investigation + recovery handling per confirmed loss
# Legitimate payment auto-blocked: remediation, late-payment penalty, supplier friction.
C_FP_RATE = 0.004         # late-payment penalty as a share of invoice value
C_FP_FIXED = 95.0         # USD, dual-control re-approval and supplier contact
# Analyst review: fully-loaded hourly cost x mean handling time.
C_REVIEW = 14.30          # USD per item routed to the manual queue


@dataclass(frozen=True)
class Costs:
    fn_rate: float
    fn_fixed: float
    fp_rate: float
    fp_fixed: float
    review: float


# A held B2B invoice and a declined card authorisation are not the same event and
# must not carry the same price. The invoice book is a five-figure median ticket with
# a dual-control release path; the card portfolio is an EUR 88 mean ticket where the
# whole remediation is a re-auth. Costing the card stream at B2B rates makes a 6%
# hold rate look catastrophic for arithmetic reasons that have nothing to do with the
# method, so each stream carries its own constants.
STREAM_COSTS = {
    "synb2b": Costs(0.72, 180.0, 0.004, 95.0, 14.30),
    "ulb": Costs(0.91, 11.50, 0.0, 4.60, 2.10),
}


def costs_for(stream: str) -> Costs:
    return STREAM_COSTS.get(stream, STREAM_COSTS["synb2b"])


# ----------------------------------------------------------------------------
# Endogenous label arrival (Layer B)
# ----------------------------------------------------------------------------
REVIEW_DELAY_SHAPE = 2.0
REVIEW_DELAY_SCALE = 0.4       # days; analyst adjudication
DISCLOSE_DELAY_SHAPE = 2.0
DISCLOSE_DELAY_SCALE = 30.0    # days; chargeback / reconciliation exception
DISCLOSE_DELAY_CAP = 180.0     # days; after this the ledger is closed
MATURITY_DAYS = 180.0          # allow-path items are booked "clean" at this age

# Topology-specific disclosure probability at mean invoice size. Wire redirection and
# shell rings surface reliably (the counterparty complains, the bank recall fails
# loudly); payment-term manipulation usually never surfaces at all.
P_DISCLOSE_BASE: Dict[str, float] = {
    "T5_wire_redirection": 0.86,
    "T4_shell_supplier_ring": 0.71,
    "T1_vendor_injection": 0.58,
    "T2_invoice_cycling": 0.34,
    "T3_payment_term_manipulation": 0.19,
    "D_drift_induced": 0.42,
}
P_DISCLOSE_AMOUNT_COEF = 0.62  # logit shift per unit z-score of log invoice amount
P_DISCLOSE_FLOOR = 0.03
P_DISCLOSE_CEIL = 0.97
# ULB has no topology annotation; disclosure is driven by amount alone.
P_DISCLOSE_ULB_INTERCEPT = 0.50
P_DISCLOSE_ULB_COEF = 0.75


# ----------------------------------------------------------------------------
# Drift layer (Layer A) - label-generating process
# ----------------------------------------------------------------------------
LGP_GAMMA_AMT = 0.55  # amount effect, held fixed across the stream
LGP_BETA_BASE = 0.90  # novelty effect outside the concept-shift regime

# a0, beta_flipped and lambda_peak are solved per stream against stated targets by
# tools/calibrate_drift.py (`make drift-params`) and versioned in drift_params.json,
# so the same three events mean the same thing on books with different base rates.
_PARAMS_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "drift_params.json")
with open(_PARAMS_PATH) as _f:
    DRIFT_PARAMS: Dict[str, dict] = _json.load(_f)

# Fallback values, used only when a stream is absent from drift_params.json.
LGP_A0 = -7.4284
LGP_BETA_FLIPPED = -4.9533
LGP_LAMBDA_PEAK = 6.8394


def lgp_params(stream: str):
    p = DRIFT_PARAMS.get(stream)
    if p is None:
        return LGP_A0, LGP_BETA_FLIPPED, LGP_LAMBDA_PEAK
    return p["a0"], p["beta_flipped"], p["lambda_peak"]

DRIFT_FRACTIONS = (0.28, 0.55, 0.80)   # positions within the evaluation stream
COVARIATE_LOGAMT_SHIFT = 0.55          # persistent upward shift in log invoice amount
COVARIATE_RAMP_DAYS = 3.0
CONCEPT_DURATION_DAYS = 60.0
PRIOR_RAMP_DAYS = 4.0
PRIOR_HALFLIFE_DAYS = 9.0
PRIOR_TAIL_DAYS = 40.0


# ----------------------------------------------------------------------------
# Policy defaults
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyConfig:
    alpha: float = 0.0124          # target false-omission rate on the auto-allow path
    budget: float = 0.0200         # hard cap on the share of traffic sent to review
    block_cap: float = 0.0800      # operational cap on the hard-hold share
    epsilon: float = 0.12          # share of the review budget spent on exploration
    gamma_aci: float = 0.0180      # ACI step size
    rho: float = 0.9975            # per-day calibration decay (half-life 277 d)
    window_max_n: int = 45000      # cap on calibration window length (items)
    window_max_days: float = 240.0
    update_every: int = 250        # transactions between threshold updates
    min_calib_items: int = 120     # below this the thresholds simply do not move
    min_support: float = 200.0     # calibration weight a threshold must stand on
    pi_floor: float = 0.05         # drop calibration items below this arrival propensity
    weight_clip: float = 40.0      # cap on any single inverse-propensity weight
    metric_block: int = 2000       # transactions per reporting window

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    stream: str = "synb2b"
    method: str = "M5"
    seed: int = 0
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    delay_scale_mult: float = 1.0      # ablation: multiplies every arrival delay
    disclose_mult: float = 1.0         # ablation: multiplies p_disclose (1.0 = as-is)
    drift_severity: float = 1.0        # ablation: scales every drift magnitude
    small: bool = False                # reduced-size pass for `make reproduce-small`


METHODS = ["M0", "M1", "M2", "M3", "M4", "M5"]
METHOD_LABELS = {
    "M0": "STATIC",
    "M1": "CONF-FIXED",
    "M2": "COST-THRESH",
    "M3": "ACI-BUDGET",
    "M4": "ACI-BUDGET-CRC",
    "M5": "ACI-BUDGET-CRC-IAP",
}

TOPOLOGIES = [
    "T1_vendor_injection",
    "T2_invoice_cycling",
    "T3_payment_term_manipulation",
    "T4_shell_supplier_ring",
    "T5_wire_redirection",
]

# Risk targets are set per stream because the two books do not share a base rate: the
# B2B invoice book runs at 2.14% fraud in the quiet regime, the card portfolio at
# 0.163%. Each target sits modestly above its own stream's Proposition-1 risk floor,
# which is how an operations team picks one - you commit to a number you can hold in a
# quiet quarter, and you find out during a bad one whether that was optimistic.
STREAM_ALPHA = {"synb2b": 0.0124, "ulb": 0.00093}


def policy_for(stream: str, **overrides) -> "PolicyConfig":
    kw = {"alpha": STREAM_ALPHA.get(stream, 0.0155)}
    kw.update(overrides)
    return PolicyConfig(**kw)


TRAIN_FRACTION = 0.20
BOOTSTRAP_RESAMPLES = 10000
SEEDS = [11, 23, 37, 53, 71]
