// Typed client for the SENTINEL FastAPI backend (the model lives there).
// Same-origin by default: in production FastAPI serves both the UI and the API, so a
// bare path like "/summary" just works. In dev, Vite proxies those paths to :8000.
// Override with VITE_API_URL to point at a separate backend host.
const BASE = (import.meta as any).env?.VITE_API_URL ?? "";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

export interface Driver {
  feature: string; label?: string; value: any; value_readable?: string;
  shap: number; direction: string; context?: string;
}
export interface AccountRow {
  account_id: number; risk_score: number; probability: number; band: string;
  confidence_tier: string; ground_truth: string | null;
}
export interface AccountDetail extends AccountRow {
  top_drivers: Driver[]; report: string;
  data_quality: { completeness: number; n_blank: number; n_features: number; model_trust: number };
}
export interface Summary {
  headline: {
    pr_auc: number; pr_std: number; roc_auc: number; brier: number; naive_with_leak: number;
    bootstrap_ci: [number, number] | null; oof_pr_auc: number; infold_pr_auc: number; bucket_leaks_removed: number;
  };
  leakage_sweep: { leak_thr: number; n_leaks: number; n_feats: number; pr_auc: number; roc_auc: number }[];
  abstention: any; decisioning: any; realtime: any; anomaly: any; source: string;
}
export interface RingsResp {
  network: { nodes: { id: number; risk: number }[]; edges: { source: number; target: number; weight: number }[] } | null;
  rings: { rings: any[]; n_candidate_rings: number; mules_in_rings: number; n_mules: number; validation: any } | null;
}
export interface FlowResp {
  typologies: { type: string; n_accounts: number; accounts: number[];
    edges: { source: number; target: number; amount: number }[]; total_amount: number }[];
  note?: string;
}

export interface NetResp {
  account_id: number;
  nodes: { id: number; risk: number; band: string; center: boolean }[];
  edges: { source: number; target: number; weight: number }[];
  features_used: number; note: string;
}

export const api = {
  summary: () => get<Summary>("/summary"),
  accounts: () => get<{ accounts: AccountRow[]; source: string }>("/accounts"),
  account: (id: number) => get<AccountDetail>(`/account/${id}`),
  rings: () => get<RingsResp>("/rings"),
  flow: () => get<FlowResp>("/flow"),
  net: (id: number) => get<NetResp>(`/network/${id}`),
  health: () => get<{ status: string }>("/health"),
  upload: async (f: File) => {
    const fd = new FormData(); fd.append("file", f);
    const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `upload -> ${r.status}`);
    return r.json() as Promise<{ ok: boolean; source: string; n_accounts: number; n_labelled_mules: number; has_labels: boolean }>;
  },
  reset: async () => { const r = await fetch(`${BASE}/reset`, { method: "POST" }); return r.json(); },
};
