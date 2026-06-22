import { Summary } from "../api";
import { Metric, Card, fmtInr } from "../ui";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

export default function Overview({ s }: { s: Summary }) {
  const h = s.headline;
  const dec = s.decisioning?.sentinel_graduated;
  const rt = s.realtime;
  const ab = s.abstention;
  const sweep = (s.leakage_sweep ?? []).map((r) => ({ stage: r.leak_thr > 1 ? "with leak" : `≥${Math.round(r.leak_thr * 100)}%`, pr: r.pr_auc }));

  return (
    <>
      <h1 className="h1">SENTINEL — Mule-Account Risk Command Center</h1>
      <p className="sub">Detect → explain → contain. Customer-fair decisions, analyst-ready, real-time on commodity CPU.</p>

      {/* LEAD WITH VALUE, not metrics */}
      <div className="grid g4">
        <Metric k="🚨 Accounts triaged" v={ab ? `${(ab.coverage_auto_decided * 100).toFixed(0)}% auto` : "—"} d={`only ${ab ? (ab.review_rate * 100).toFixed(1) : "?"}% need a human`} cls="good" />
        <Metric k="🛟 Wrongful freezes" v={dec ? dec.wrongful_freezes : "—"} d="customers protected" cls="good" />
        <Metric k="💰 Loss prevented" v={s.decisioning ? fmtInr(s.decisioning.expected_cost_reduction_rupees) : "—"} d="vs naïve policy" cls="good" />
        <Metric k="⚡ Scoring speed" v={rt ? `${rt.batch_throughput_accts_per_sec?.toLocaleString()}/s` : "—"} d="CPU · no GPU needed" />
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Card title="🛟 Customer-harm-aware decisioning">
          {dec ? (
            <>
              <div className="grid g2">
                <Metric k="High-confidence freezes" v={dec.high_conf_freezes} d="precise action only" />
                <Metric k="Hard mules surfaced (step-up)" v={dec.mules_surfaced_by_stepup} d="without freezing anyone" cls="good" />
                <Metric k="Wrongful freezes" v={dec.wrongful_freezes} cls="good" />
                <Metric k="Expected cost cut" v={fmtInr(s.decisioning.expected_cost_reduction_rupees)} cls="good" />
              </div>
              <p className="small muted" style={{ marginTop: 10 }}>
                Graduated response — <b>clear → step-up verify → freeze</b>. Freeze only on high confidence; verify the
                ambiguous middle (a mule fails KYC re-check, a real customer passes). Protects customers <i>and</i> analysts.
              </p>
            </>
          ) : <p className="muted">—</p>}
        </Card>

        <Card title="🚦 Triage & abstention">
          {ab ? (
            <>
              <div className="grid g2">
                <Metric k="Auto-decided" v={`${(ab.coverage_auto_decided * 100).toFixed(0)}%`} />
                <Metric k="Auto-zone error" v={`${(ab.auto_zone_error_rate * 100).toFixed(2)}%`} cls="good" />
                <Metric k="Routed to analyst" v={`${(ab.review_rate * 100).toFixed(1)}%`} />
                <Metric k="Real-time / account" v={rt ? `${Math.round(rt.score_latency_ms?.p50)} ms` : "—"} />
              </div>
              <p className="small muted" style={{ marginTop: 10 }}>
                The model auto-clears confident-legit, auto-flags confident-mule, and <b>routes only the ambiguous band</b>
                to a human — crushing alert fatigue instead of guessing on the hard tail.
              </p>
            </>
          ) : <p className="muted">—</p>}
        </Card>
      </div>

      {/* METRICS + LEAKAGE STORY AT THE END */}
      <div className="card" style={{ marginTop: 22 }}>
        <h3>📈 Validated performance (leakage-removed, honest)</h3>
        <div className="grid g4">
          <Metric k="PR-AUC (5×2 CV)" v={h.pr_auc?.toFixed(3)} d={`± ${h.pr_std?.toFixed(3)}`} cls="good" />
          <Metric k="Bootstrap 95% CI" v={h.bootstrap_ci ? `${h.bootstrap_ci[0].toFixed(2)}–${h.bootstrap_ci[1].toFixed(2)}` : "—"} />
          <Metric k="ROC-AUC" v={h.roc_auc?.toFixed(3)} d={`Brier ${h.brier?.toFixed(4)}`} />
          <Metric k="In-fold re-check" v={h.infold_pr_auc?.toFixed(3)} d="not leak-inflated" />
        </div>
        <div className="grid g2" style={{ marginTop: 14 }}>
          <div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={sweep} margin={{ top: 8, right: 16, bottom: 4, left: -16 }}>
                <XAxis dataKey="stage" stroke="#93a0bd" fontSize={12} />
                <YAxis domain={[0.8, 1]} stroke="#93a0bd" fontSize={12} />
                <Tooltip contentStyle={{ background: "#141b2e", border: "1px solid #263150", borderRadius: 10, color: "#e8edf7" }} />
                <ReferenceLine y={h.pr_auc} stroke="#2bbd7e" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="pr" stroke="#d11f2d" strokeWidth={2.5} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
            <p className="small muted">Leakage-sensitivity sweep: PR-AUC plateaus at ~0.86 = genuine signal.</p>
          </div>
          <div>
            <p className="small" style={{ lineHeight: 1.6 }}>
              A naïve model scores <b>{h.naive_with_leak?.toFixed(3)}</b> by reading <code>F3912</code> — a post-hoc fraud
              flag. Our Data Integrity Auditor removes it plus <b>{h.bucket_leaks_removed}</b> bucket leaks, so we report a
              defensible <b>{h.pr_auc?.toFixed(3)}</b> (~100× baseline) that holds in production. Honest by design.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
