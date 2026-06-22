import { useMemo, useState } from "react";
import { Summary } from "../api";
import { Card } from "../ui";

// Knowledge-base copilot: answers questions about how SENTINEL works + common fraud/AML
// concepts, grounded in the live numbers. Runs fully in-browser (no API key). To enable
// open-domain answers, point it at an LLM endpoint — but this covers the demo Q&A offline.
type QA = { k: string[]; a: (s: Summary) => string };

const KB: QA[] = [
  { k: ["how", "work", "pipeline", "overview", "what is sentinel", "about"], a: () =>
    "SENTINEL scores each account 0–100 for mule-risk using a calibrated LightGBM model, explains every alert with SHAP, groups look-alike accounts into candidate rings, and recommends a graduated action (clear → step-up verify → freeze). It abstains on ambiguous cases and routes them to a human." },
  { k: ["leak", "f3912", "0.99", "0.998", "honest", "why low"], a: (s) =>
    `The raw data has target leakage: feature F3912 is a post-hoc fraud flag ~96% aligned with the label, so a naïve model scores PR-AUC ${s.headline.naive_with_leak?.toFixed(3)} by reading the answer. Our Data Integrity Auditor removes it plus ${s.headline.bucket_leaks_removed} bucket leaks, giving a defensible PR-AUC ${s.headline.pr_auc?.toFixed(3)} that holds in production.` },
  { k: ["pr-auc", "prauc", "accuracy", "metric", "performance", "roc", "score how good"], a: (s) =>
    `PR-AUC ${s.headline.pr_auc?.toFixed(3)} (±${s.headline.pr_std?.toFixed(3)}, 5×2 CV), bootstrap 95% CI ${s.headline.bootstrap_ci ? s.headline.bootstrap_ci.join("–") : "—"}, ROC-AUC ${s.headline.roc_auc?.toFixed(3)}, Brier ${s.headline.brier?.toFixed(4)}. We lead with PR-AUC (not accuracy) because at ~1% fraud, "predict all clean" is 99% accurate but useless.` },
  { k: ["ring", "network", "cluster", "graph", "color", "colour", "legend"], a: (s) =>
    `We group mules into candidate rings via a behavioural-similarity graph (${s.realtime ? "" : ""}accounts that look near-identical on genuine features). In the maps, node colour = risk band: red ≥90, amber 70–89, blue 40–69, green <40; node size ∝ risk; the white-ringed node is the account you selected. It's a proxy for the link data a bank holds — confirmation needs device/beneficiary data (Phase-2).` },
  { k: ["abstain", "uncertain", "confidence", "route", "review"], a: (s) =>
    `Instead of forcing a decision on every account, SENTINEL auto-decides ${s.abstention ? (s.abstention.coverage_auto_decided * 100).toFixed(0) : "~99"}% at ${s.abstention ? (s.abstention.auto_zone_error_rate * 100).toFixed(2) : "~0.1"}% error and routes the ambiguous ${s.abstention ? (s.abstention.review_rate * 100).toFixed(1) : "~0.7"}% to a human — so analysts only see the cases that matter.` },
  { k: ["freeze", "customer", "harm", "decision", "action", "step-up", "stepup", "contain"], a: (s) =>
    `Graduated response: clear → step-up verify → freeze. We freeze only high-confidence mules and step-up-verify the ambiguous middle (a mule fails KYC re-check, a real customer passes). On the sample that's ${s.decisioning?.sentinel_graduated?.wrongful_freezes ?? 0} wrongful freezes and ${s.decisioning?.sentinel_graduated?.mules_surfaced_by_stepup ?? "several"} hard mules surfaced — protecting customers and analysts.` },
  { k: ["latency", "fast", "real-time", "realtime", "speed", "gpu", "ms"], a: (s) =>
    `Real-time on commodity CPU — ${s.realtime ? s.realtime.batch_throughput_accts_per_sec?.toLocaleString() : "~2,300"} accounts/sec batch (~0.4 ms each), on-demand ~${s.realtime ? Math.round(s.realtime.score_latency_ms?.p50) : "200"} ms. GPUs give no benefit on tabular boosting, so there's no GPU dependency — easier for a bank to deploy.` },
  { k: ["anomaly", "isolation forest", "outlier", "lof", "unsupervised"], a: (s) =>
    `We tested unsupervised anomaly detection (Isolation Forest + LOF). Finding: mules here are NOT statistical outliers — both score near random (${s.anomaly ? s.anomaly.anomaly_pr_auc?.toFixed(3) : "~0.01"}), and a naïve hybrid would drop the score. So this is a supervised problem; we report that honestly rather than ship a metric that hurts.` },
  { k: ["upload", "csv", "new data", "dataset", "my data"], a: () =>
    "Use ⬆ Upload account CSV (top bar). The whole dashboard re-scores on your file — Investigate, Network, Alerts and Analytics all update. Reset returns to the bank sample. Uploaded data is processed in-session, not stored." },
  { k: ["alert", "threshold", "queue", "manage", "accountability", "sla"], a: () =>
    "Alert Management turns every account above the alert threshold into a case with a status (New → Investigating → Escalated → Cleared) and an owner. Drag the threshold to trade recall for workload. It works on the sample or any uploaded dataset, giving an auditable trail." },
  { k: ["mule", "what is a mule", "mule account"], a: () =>
    "A mule account is one used to receive and move illicit funds — often a real customer's account that's been recruited or compromised. Detecting mules early breaks the money-laundering chain before funds are cashed out." },
  { k: ["data", "features", "how many", "accounts", "imbalance"], a: (s) =>
    "The model uses ~2,965 leak-removed features per account (anonymized). Mules are ~0.9% of accounts — severe imbalance, which is exactly why PR-AUC, abstention and graduated action matter more than raw accuracy." },
  { k: ["deploy", "production", "bank", "integrate"], a: () =>
    "Deployment is API-first: a FastAPI service hosts the model (/score, /report, /network) and the React UI. It runs on CPU, so it slots into existing bank infra — no GPU fleet. Auth, rate-limiting and a durable case store are built in." },
];

const SUGGESTED = ["How does SENTINEL work?", "Why is your PR-AUC not 0.99?", "What do the ring colours mean?",
  "How do you protect innocent customers?", "Is it real-time?", "Did you test anomaly detection?", "How do I upload my data?"];

function answer(q: string, s: Summary): string {
  const t = q.toLowerCase();
  let best: QA | null = null, score = 0;
  for (const e of KB) { const m = e.k.filter((kw) => t.includes(kw)).length; if (m > score) { score = m; best = e; } }
  if (best && score > 0) return best.a(s);
  return "I can answer questions about how SENTINEL works — the model, the leakage story, rings, abstention, customer-fair decisions, real-time performance, anomaly detection, uploading data, and deployment. Try one of the suggested questions above.";
}

export default function Copilot({ s }: { s: Summary }) {
  const [q, setQ] = useState("");
  const [log, setLog] = useState<{ q: string; a: string }[]>([
    { q: "", a: "👋 I'm the SENTINEL copilot. Ask me how the system works — leakage, rings, decisions, real-time, anything about the prototype." },
  ]);
  const ask = (text: string) => { if (!text.trim()) return; setLog((l) => [...l, { q: text, a: answer(text, s) }]); setQ(""); };
  const chips = useMemo(() => SUGGESTED, []);
  return (
    <>
      <h1 className="h1">AI Copilot</h1>
      <p className="sub">Ask anything about how SENTINEL works — grounded in the live numbers. Runs in-browser.</p>
      <div className="row" style={{ gap: 8, marginBottom: 14, flexWrap: "wrap" }}>
        {chips.map((c) => <button key={c} className="ghost" style={{ fontSize: 12.5 }} onClick={() => ask(c)}>{c}</button>)}
      </div>
      <Card>
        <div style={{ maxHeight: 420, overflow: "auto", display: "flex", flexDirection: "column", gap: 12 }}>
          {log.map((m, i) => (
            <div key={i}>
              {m.q && <div style={{ textAlign: "right" }}><span className="pill" style={{ background: "var(--accent)", color: "#fff", display: "inline-block" }}>{m.q}</span></div>}
              <div style={{ marginTop: m.q ? 8 : 0 }}><span className="pill" style={{ display: "inline-block", lineHeight: 1.55 }}>🤖 {m.a}</span></div>
            </div>
          ))}
        </div>
        <div className="row" style={{ marginTop: 14, gap: 8 }}>
          <input style={{ flex: 1 }} placeholder="Ask the copilot…" value={q}
            onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ask(q)} />
          <button className="primary" onClick={() => ask(q)}>Ask</button>
        </div>
      </Card>
      <p className="small muted" style={{ marginTop: 10 }}>Answers come from a curated knowledge base of the system (offline, no external calls). It can be wired to an LLM endpoint for open-domain Q&A.</p>
    </>
  );
}
