"""
SENTINEL — Phase-2 INGESTION layer: suspicious-transaction detection + alert/ticket fusion.

PS2 asks for a system that ingests financial transactions and fraud/TMS/govt alert feeds and
prevents circulation of fraudulent proceeds. The provided BOI data is an account *snapshot*
(no transactions/alerts), so this module is the working ingestion engine that runs on ANY
standard transaction or alert feed a bank would plug in — demonstrated on uploaded feeds.

It is RULE-BASED and explainable (how real transaction-monitoring systems work): no training
data needed, every flag carries its reasons. Two entry points:

  score_transactions(df)  -> per-transaction suspicion (0-100) + reasons + per-account rollup
  fuse_alerts(alerts, account_risk) -> corroborate external alert tickets against model risk
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# flexible column detection so it works on PaySim-style and generic bank exports
def _find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def score_transactions(df: pd.DataFrame) -> dict:
    """Score each transaction for suspicion using TMS-style typology rules. Returns flagged
    transactions (with reasons) and an account-level rollup (the circulation view)."""
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    amt = _find(df, "amount", "amt", "txn_amount", "tx_amount", "transactionamount")
    typ = _find(df, "type", "txn_type", "tx_type", "transactiontype")
    ob = _find(df, "oldbalanceorg", "oldbalanceorig", "orig_old_balance")
    nb = _find(df, "newbalanceorig", "newbalanceorg", "orig_new_balance")
    obd = _find(df, "oldbalancedest", "dest_old_balance")
    nbd = _find(df, "newbalancedest", "dest_new_balance")
    orig = _find(df, "nameorig", "sender", "sender_account_id", "orig", "from_account")
    dest = _find(df, "namedest", "receiver", "receiver_account_id", "dest", "to_account")
    if amt is None:
        raise ValueError("No amount column found (expected one of: amount, amt, txn_amount, tx_amount).")
    a = pd.to_numeric(df[amt], errors="coerce").fillna(0.0)

    score = np.zeros(len(df)); reasons = [[] for _ in range(len(df))]

    def add(mask, pts, why):
        mask = np.asarray(mask)
        score[mask] += pts
        for i in np.where(mask)[0]:
            reasons[i].append(why)

    # 1) high-value spike (top 1% by amount)
    hi = a >= np.quantile(a[a > 0], 0.99) if (a > 0).any() else a > np.inf
    add(hi, 30, "high-value (top 1% by amount)")
    # 2) structuring — amounts just under a round reporting threshold (classic smurfing)
    for thr in (10000, 50000, 100000, 200000):
        add((a >= 0.9 * thr) & (a < thr), 25, f"structuring (just under {thr:,})")
    # 3) account-drain — origin emptied to ~0
    if ob and nb:
        o = pd.to_numeric(df[ob], errors="coerce").fillna(0); n = pd.to_numeric(df[nb], errors="coerce").fillna(0)
        add((n == 0) & (o > 0), 25, "account drained to zero")
        add((np.abs(n + a - o) > 1) & (o > 0), 15, "origin balance inconsistency")
    # 4) destination balance not credited (pass-through mule)
    if obd and nbd:
        od = pd.to_numeric(df[obd], errors="coerce").fillna(0); nd = pd.to_numeric(df[nbd], errors="coerce").fillna(0)
        add((np.abs(od + a - nd) > 1) & (a > 0), 15, "destination balance inconsistency (pass-through)")
    # 5) cash-out / transfer typology
    if typ:
        tl = df[typ].astype(str).str.upper()
        add(tl.isin(["CASH_OUT", "TRANSFER", "WITHDRAWAL"]).values, 10, "cash-out/transfer channel")
    # 6) TRANSFER -> CASH_OUT chain (money in via transfer, out via cash-out on the same account)
    if typ and orig and dest:
        t_dest = set(df.loc[df[typ].astype(str).str.upper() == "TRANSFER", dest])
        c_orig = set(df.loc[df[typ].astype(str).str.upper() == "CASH_OUT", orig])
        chain = t_dest & c_orig
        in_chain = df[orig].isin(chain).values | df[dest].isin(chain).values
        add(in_chain, 20, "layering chain (transfer-in then cash-out)")
    # 7) velocity — origin account with many transactions in the file (rapid movement)
    if orig:
        vc = df.groupby(orig)[amt].transform("count")
        add((vc >= 5).values, 10, "high velocity (>=5 txns by this account)")

    score = np.clip(score, 0, 100)
    out = df.copy()
    out["suspicion_score"] = score.astype(int)
    out["reasons"] = ["; ".join(r) if r else "—" for r in reasons]
    out["flag"] = out["suspicion_score"] >= 50
    # account-level rollup (circulation view): worst + total suspicious value per origin account
    rollup = pd.DataFrame()
    if orig:
        g = out.groupby(df[orig])
        rollup = pd.DataFrame({
            "account": g.size().index,
            "txns": g.size().values,
            "max_suspicion": g["suspicion_score"].max().values,
            "suspicious_txns": g.apply(lambda x: int((x["suspicion_score"] >= 50).sum())).values,
            "suspicious_value": g.apply(lambda x: float(x.loc[x["suspicion_score"] >= 50, amt].sum())).values,
        }).sort_values("max_suspicion", ascending=False)
    return {"transactions": out, "rollup": rollup, "n_flagged": int(out["flag"].sum()),
            "amount_col": amt, "orig_col": orig}


def fuse_alerts(alerts: pd.DataFrame, account_risk: pd.Series | None = None) -> dict:
    """Ingest external alert tickets (TMS / fraud-monitoring / govt cyber-fraud) and corroborate
    against the model's account risk. An account flagged by BOTH an external feed AND the model
    is the highest-priority, most-defensible escalation."""
    alerts = alerts.copy(); alerts.columns = [str(c).strip() for c in alerts.columns]
    acc = _find(alerts, "account", "account_id", "acct", "accountid", "nameorig", "customer_id")
    src = _find(alerts, "source", "feed", "system")
    sev = _find(alerts, "severity", "priority")
    if acc is None:
        raise ValueError("Alert feed needs an account column (account / account_id / acct).")
    alerts["_account"] = pd.to_numeric(alerts[acc], errors="coerce")
    alerts["_source"] = alerts[src] if src else "external"
    alerts["_severity"] = alerts[sev] if sev else "—"
    corrob = []
    for _, r in alerts.iterrows():
        a = r["_account"]
        mr = None if account_risk is None or pd.isna(a) or a not in account_risk.index else int(account_risk.loc[a])
        corrob.append({"account": (int(a) if pd.notna(a) else None), "source": r["_source"],
                       "severity": r["_severity"], "model_risk": mr,
                       "status": ("CORROBORATED — escalate" if (mr is not None and mr >= 70)
                                  else "model also elevated" if (mr is not None and mr >= 40)
                                  else "model low / unknown — review")})
    cdf = pd.DataFrame(corrob)
    n_corrob = int((cdf["status"].astype(str).str.startswith("CORROBORATED")).sum()) if len(cdf) else 0
    return {"alerts": cdf, "n_alerts": len(cdf), "n_corroborated": n_corrob,
            "by_source": (cdf["source"].value_counts().to_dict() if len(cdf) else {})}


def sample_transactions(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """A tiny SYNTHETIC transaction feed (clearly labelled) so the ingestion layer can be
    demoed without a real feed. NOT used for any model training or reported metric."""
    rng = np.random.RandomState(seed)
    types = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"], n, p=[0.5, 0.2, 0.2, 0.1])
    amt = rng.lognormal(9, 1.2, n).round(2)
    # plant a few obvious typologies
    amt[:5] = [49500, 49000, 99000, 9500, 195000]            # structuring
    old = rng.lognormal(10, 1, n).round(2)
    new = np.maximum(old - amt, 0)
    new[:5] = 0                                              # drains
    return pd.DataFrame({
        "step": rng.randint(1, 30, n), "type": types, "amount": amt,
        "nameOrig": [f"A{rng.randint(9000, 9082)}" for _ in range(n)],
        "oldbalanceOrg": old, "newbalanceOrig": new,
        "nameDest": [f"M{rng.randint(1, 60)}" for _ in range(n)],
        "oldbalanceDest": rng.lognormal(8, 1, n).round(2), "newbalanceDest": rng.lognormal(8, 1, n).round(2),
    })
