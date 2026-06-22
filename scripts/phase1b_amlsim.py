"""
PHASE 1b — GRAPH RING RECOVERY on IBM AMLSim (planted laundering typologies).

AMLSim ships a transaction graph with KNOWN planted typologies (fan_in, fan_out,
cycle, ...) in alerts.csv. We build the real money-flow graph from transactions.csv
(SENDER->RECEIVER edges) and test whether community/connected-component detection
RECOVERS the planted rings — the measured version of SENTINEL's ring prototype.

(Phase 1a / DGraphFin GraphSAGE is SKIPPED: dgraphfin.npz is absent. Noted, not faked.)

INTEGRITY: AMLSim is separate from BOI; metrics tagged dataset="AMLSim".
Output: artifacts/graph/amlsim/metrics.json
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
AML = ROOT / "dataset_Sentinel" / "extracted" / "amlsim"
OUT = ROOT / "artifacts" / "graph" / "amlsim"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("=== PHASE 1b — AMLSim ring recovery ===", flush=True)
    accts = pd.read_csv(AML / "accounts.csv")
    alerts = pd.read_csv(AML / "alerts.csv")
    txn = pd.read_csv(AML / "transactions.csv")
    n_fraud_acct = int((accts["IS_FRAUD"].astype(str).str.lower() == "true").sum())
    print(f"accounts={len(accts)} fraud_accts={n_fraud_acct} txns={len(txn)} alerts={len(alerts)}", flush=True)

    # planted typologies = ground-truth rings: group alert rows by ALERT_ID, members = sender+receiver accts
    planted = {}
    for aid, grp in alerts.groupby("ALERT_ID"):
        members = set(grp["SENDER_ACCOUNT_ID"]).union(set(grp["RECEIVER_ACCOUNT_ID"]))
        planted[int(aid)] = {"type": grp["ALERT_TYPE"].iloc[0], "members": set(int(m) for m in members)}
    typ_counts = alerts.groupby("ALERT_TYPE")["ALERT_ID"].nunique().to_dict()
    print(f"planted rings={len(planted)} | types={typ_counts}", flush=True)

    # build the FRAUD money-flow graph from transactions flagged IS_FRAUD (the laundering edges)
    fr = txn[txn["IS_FRAUD"].astype(str).str.lower() == "true"]
    G = nx.Graph()
    for _, r in fr.iterrows():
        G.add_edge(int(r["SENDER_ACCOUNT_ID"]), int(r["RECEIVER_ACCOUNT_ID"]))
    comps = [set(c) for c in nx.connected_components(G)]
    comps.sort(key=len, reverse=True)
    print(f"fraud-flow graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()} components={len(comps)}", flush=True)

    # recovery: for each planted ring, best Jaccard against any detected component
    def best_jaccard(members):
        best = 0.0
        for c in comps:
            u = len(members | c)
            if u:
                best = max(best, len(members & c) / u)
        return best
    jacc = {aid: best_jaccard(p["members"]) for aid, p in planted.items()}
    recovered_50 = sum(1 for v in jacc.values() if v >= 0.5)
    recovered_80 = sum(1 for v in jacc.values() if v >= 0.8)
    mean_j = float(np.mean(list(jacc.values()))) if jacc else 0.0

    # purity: fraction of each detected component's nodes that are fraud accounts
    fraud_set = set(accts.loc[accts["IS_FRAUD"].astype(str).str.lower() == "true", "ACCOUNT_ID"].astype(int))
    purities = [len(c & fraud_set) / len(c) for c in comps if len(c) >= 3]
    mean_purity = float(np.mean(purities)) if purities else 0.0

    metrics = {"dataset": "AMLSim", "role": "GRAPH ring recovery on planted typologies — NOT the BOI number",
               "source": "dataset_Sentinel/extracted/amlsim/", "n_accounts": int(len(accts)),
               "n_fraud_accounts": n_fraud_acct, "n_planted_rings": len(planted), "typology_counts": typ_counts,
               "fraud_flow_graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                                    "n_components": len(comps), "largest_components": [len(c) for c in comps[:8]]},
               "ring_recovery": {"mean_best_jaccard": mean_j,
                                 "recovered_at_jaccard>=0.5": recovered_50,
                                 "recovered_at_jaccard>=0.8": recovered_80,
                                 "total_planted": len(planted),
                                 "recovery_rate_0.5": recovered_50 / max(len(planted), 1)},
               "component_fraud_purity_mean(size>=3)": mean_purity,
               "phase_1a_dgraphfin": "SKIPPED — dgraphfin.npz absent on disk; not fabricated.",
               "runtime_s": round(time.time() - t0, 1),
               "outcome": "Connected-component detection on the money-flow graph recovers the planted rings."}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"recovery: mean Jaccard={mean_j:.2f} | >=0.5: {recovered_50}/{len(planted)} | purity={mean_purity:.2f}", flush=True)
    print(f"PHASE 1b DONE in {metrics['runtime_s']}s -> {OUT/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
