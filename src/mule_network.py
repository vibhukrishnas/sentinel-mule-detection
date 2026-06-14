"""
SENTINEL — mule-ring detection PROTOTYPE (behavioral-similarity graph).

HONEST framing: the provided data is an account snapshot with NO explicit link data
(no shared device / IP / beneficiary / transaction edges). So we build a
behavioral-similarity graph: an edge connects two mules that are near-identical on the
genuine (leak-removed) behavioral features — a data-grounded PROXY for the links a bank
would have. Connected components = candidate mule RINGS to investigate together.

With real shared-device / beneficiary / transaction edges (Phase-2) this same engine
becomes production mule-network detection. Output: figures/11_mule_network.png + JSON.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from preprocess import load_cached, ART

FIG = ART.parent / "figures"
RED, NAVY, GREY = "#d11f2d", "#1f3b6e", "#8a8a8a"
PALETTE = ["#d11f2d", "#1f77b4", "#2ca02c", "#9467bd", "#e8a33d", "#17becf", "#e377c2", "#8c564b"]


def main():
    X, y = load_cached()
    mules = X[y == 1]
    print(f"Building behavioral-similarity graph over {len(mules)} mules...", flush=True)

    # use the model's top genuine features (interpretable + signal-bearing)
    base = joblib.load(ART / "base_model.joblib")
    gain = pd.Series(base.booster_.feature_importance("gain"), index=X.columns)
    topf = gain.sort_values(ascending=False).head(30).index.tolist()

    M = mules[topf]
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(M))
    S = cosine_similarity(Z)
    np.fill_diagonal(S, 0.0)

    # edge if similarity is in the top ~6% of all mule-pairs (tight behavioural twins)
    thr = np.percentile(S[np.triu_indices_from(S, k=1)], 94)
    ids = list(mules.index)
    G = nx.Graph(); G.add_nodes_from(ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] >= thr:
                G.add_edge(ids[i], ids[j], w=float(S[i, j]))

    comps = [c for c in nx.connected_components(G) if len(c) >= 3]   # rings = clusters of >=3
    comps.sort(key=len, reverse=True)
    in_rings = sum(len(c) for c in comps)
    paired = sum(1 for n in G.nodes if G.degree(n) >= 1)
    print(f"edges={G.number_of_edges()} | mules with >=1 tie={paired}/{len(ids)} | "
          f"candidate rings (>=3)={len(comps)} | mules in rings={in_rings}")
    if comps:
        print(f"largest ring: {len(comps[0])} mules -> accounts {sorted(comps[0])[:8]}...")

    # ---- per-ring detail for the case study (real risk scores, honest exposure) ----
    AVG_MULE_LOSS = 250_000   # same assumption as insights.py; configurable
    try:
        scores = pd.read_csv(ART.parent / "outputs" / "predictions.csv").set_index("account_id")["risk_score"]
    except (FileNotFoundError, KeyError):
        scores = pd.Series(dtype=float)
    rings = []
    for ci, c in enumerate(comps):
        members = sorted(int(n) for n in c)
        sc = {int(a): int(scores.get(a)) for a in members if a in scores.index}
        rep = max(sc, key=sc.get) if sc else members[0]
        rings.append({"ring_id": ci + 1, "size": len(members), "members": members,
                      "rep_account": rep, "rep_score": sc.get(rep),
                      "exposure_rupees": len(members) * AVG_MULE_LOSS})
        print(f"  ring #{ci+1}: {len(members)} accts | rep #{rep} (risk {sc.get(rep)}) | "
              f"exposure ~Rs {len(members)*AVG_MULE_LOSS:,}")

    # ---- visualize ----
    plt.figure(figsize=(9, 6.2), dpi=130)
    pos = nx.spring_layout(G, seed=42, k=0.45)
    member = {}
    for ci, c in enumerate(comps):
        for n in c:
            member[n] = ci
    node_colors = [PALETTE[member[n] % len(PALETTE)] if n in member else "#cfcfcf" for n in G.nodes]
    node_sizes = [220 if n in member else 70 for n in G.nodes]
    nx.draw_networkx_edges(G, pos, edge_color=GREY, alpha=0.5, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="white", linewidths=0.5)
    plt.title(f"Mule-ring prototype: {len(comps)} candidate rings among {len(ids)} mules\n"
              f"(behavioral-similarity graph on leak-removed features - proxy for real link data)",
              fontsize=11)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(FIG / "11_mule_network.png", bbox_inches="tight"); plt.close()

    # ---- validation: is candidate Ring #1 tighter than chance, and stable? ----
    # honest backing for the Section-7 caveat: we have NO ground-truth links, so we test
    # (a) intra-ring similarity vs random legit subsets, (b) stability under feature subsampling.
    rng = np.random.RandomState(42)
    idxall = {a: i for i, a in enumerate(X.index)}
    allZ = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X[topf]))
    allS = cosine_similarity(allZ); np.fill_diagonal(allS, 0.0)
    val = {}
    if comps:
        ring1 = sorted(int(a) for a in comps[0])
        ri = [idxall[a] for a in ring1]
        sub = allS[np.ix_(ri, ri)]
        val["ring1_intra_sim"] = float(sub[np.triu_indices_from(sub, k=1)].mean())
        legit = [idxall[a] for a in X.index[y.values == 0]]
        draws = [allS[np.ix_(p, p)][np.triu_indices(len(ring1), k=1)].mean()
                 for p in (rng.choice(legit, size=len(ring1), replace=False) for _ in range(300))]
        val["legit_subset_sim_mean"] = float(np.mean(draws))
        val["legit_subset_sim_p95"] = float(np.percentile(draws, 95))
        # stability: rebuild the graph on 80% feature subsamples, Jaccard of largest component vs Ring #1
        base_set, jac, nf = set(ring1), [], len(topf)
        for _ in range(30):
            cols = rng.choice(nf, size=max(5, int(nf * 0.8)), replace=False)
            Zi = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(M.iloc[:, cols]))
            Si = cosine_similarity(Zi); np.fill_diagonal(Si, 0.0)
            ti = np.percentile(Si[np.triu_indices_from(Si, k=1)], 94)
            Gi = nx.Graph(); Gi.add_nodes_from(ids)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    if Si[a, b] >= ti:
                        Gi.add_edge(ids[a], ids[b])
            cc = set(max(nx.connected_components(Gi), key=len)) if Gi.number_of_edges() else set()
            union = len(base_set | cc)
            jac.append(len(base_set & cc) / union if union else 0.0)
        val["ring1_subsample_stability_jaccard"] = float(np.mean(jac))
        print(f"validation: Ring#1 intra-sim={val['ring1_intra_sim']:.3f} vs random legit subset "
              f"{val['legit_subset_sim_mean']:.3f} (p95 {val['legit_subset_sim_p95']:.3f}) | "
              f"feature-subsample stability Jaccard={val['ring1_subsample_stability_jaccard']:.2f}")

    out = {"n_mules": len(ids), "edges": G.number_of_edges(), "mules_with_tie": paired,
           "n_candidate_rings": len(comps), "mules_in_rings": in_rings,
           "ring_sizes": [len(c) for c in comps], "similarity_threshold": float(thr),
           "rings": rings, "validation": val, "top_features_used": topf}
    (ART / "mule_network.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"Saved figures/11_mule_network.png + artifacts/mule_network.json")


if __name__ == "__main__":
    main()
