"""
SENTINEL — Honest feature semantics for human-readable alerts.

BRUTAL-HONESTY RULE: the dataset is anonymized. We do NOT invent meanings.
  - A small set of features are *verifiable* from their values (categoricals whose
    levels are self-describing, plausible age/tenure) -> we give real labels.
  - Every other feature, including the bank's hinted "known fraud signals", is
    reported by its VALUE + POPULATION PERCENTILE + the direction SHAP says pushed
    the score. That is data-grounded and defensible in front of a judge; a fake
    semantic label is not.
"""
from __future__ import annotations

# Decoders for the few categorical features whose levels are self-describing.
F3889_TENURE = {
    "L31D": "account tenure < 31 days (very new)",
    "L90D": "account tenure < 90 days (new)",
    "L180D": "account tenure < 180 days",
    "L365D": "account tenure < 1 year",
    "G365D": "account tenure > 1 year (established)",
}

# Labels INFERRED from value patterns (no bank data dictionary was provided). Only the
# self-describing categoricals are reasonably certain; numeric ones are best-effort and
# explicitly hedged. We never present an inferred label as bank-confirmed fact.
KNOWN_LABELS = {
    "F3891": "customer occupation",                          # self-describing levels
    "F3889": "account tenure bucket",                        # self-describing codes
    "F3894": "customer age (inferred)",                      # values ~30-50
    "F3836": "large-value monetary field (inferred)",        # values up to ~5e5
    "F670":  "binary flag",
    "F2956": "0-100 bounded indicator (inferred)",
    "F1692": "low-count indicator 0-12 (inferred)",
    "F2082": "ratio 0-1 (inferred)",
    "F2122": "ratio 0-1 (inferred)",
}

# The 18 features Topic.pdf lists as "commonly used by the bank for fraud detection".
# Badge = "bank-listed" (a verifiable fact from the brief), NOT "bank-confirmed meaning".
BANK_HINTED = {"F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
               "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
               "F3889", "F3891", "F3894"}


def base_feature(col: str) -> str:
    """Strip engineered suffixes back to the source feature."""
    return col.split("__")[0]


def human_label(col: str) -> str:
    if col.endswith("__ismissing"):
        b = base_feature(col)
        tag = " [bank-listed]" if b in BANK_HINTED else ""
        return f"whether '{b}'{tag} profile field is blank"
    b = base_feature(col)
    label = KNOWN_LABELS.get(b)
    badge = " [bank-listed feature]" if b in BANK_HINTED else ""
    if label:
        return f"{label} ({b}){badge}"
    return f"behavioral indicator {b} (anonymized){badge}"


def decode_value(col: str, value) -> str:
    """Render a raw value the way an analyst would read it."""
    b = base_feature(col)
    if col.endswith("__ismissing"):
        return "BLANK" if value == 1 else "present"
    if b == "F3889" and isinstance(value, str):
        return F3889_TENURE.get(value, value)
    if b == "F670":
        return "set (1)" if value == 1 else "not set (0)"
    try:
        fv = float(value)
        return f"{fv:,.2f}" if abs(fv) >= 1 else f"{fv:.4f}"
    except (TypeError, ValueError):
        return str(value)
