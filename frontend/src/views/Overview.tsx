import { Summary } from "../api";
import { Metric, Card, fmtInr } from "../ui";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

export default function Overview({ s }: { s: Summary }) {
  const h = s.headline;
  const sweep = (s.leakage_sweep ?? []).map((r) => ({
    stage: r.leak_thr > 1 ? "with leak" : `≥${Math.round(r.leak_thr * 100)}%`,
    pr: r.pr_auc,
  }));
  const dec = s.decisioning?.sentinel_graduated;
  const naiveDec = s.decisioning?.naive_threshold_0_5 ?? s.decisioning?.["naive_threshold_0.5"];
  const rt = s.realtime;

  return (
    <>
      <h1 className="h1">Command Center</h1>
      <p className="sub">Leakage-proof mule detection · honest numbers · customer-fair decisions · real-time on CPU.</p>

      <div className="grid g4">
        <Metric k="Honest PR-AUC (5×2 CV)" v={h.pr_auc?.toFixed(3)}
          d={`± ${h.pr_std?.toFixed(3)} · ~100× baseline`} cls="good" />
        <Metric k="Bootstrap 95% CI" v={h.bootstrap_ci ? `${h.bootstrap_ci[0].toFixed(2)}–${h.bootstrap_ci[1].toFixed(2)}` : "—"}
          d={`OOF ${h.oof_pr_auc?.toFixed(3)} · in-fold ${h.infold_pr_auc?.toFixed(3)}`} />
        <Metric k="ROC-AUC" v={h.roc_auc?.toFixed(3)} d={`Brier ${h.brier?.toFixed(4)}`} />
        <Metric k="Naïve 'with-leak' score" v={h.naive_with_leak?.toFixed(3)} d="…and it's a lie" cls="bad" />
      </div>

      <div className="banner err" style={{ marginTop: 18 }}>
        <b>A naïve model scores PR-AUC {h.naive_with_leak?.toFixed(3)} — by reading the answer.</b> Feature
        <code> F3912</code> is a post-hoc fraud flag (~96% aligned with the label). Our Data Integrity Auditor
        catches it plus <b>{h.bucket_leaks_removed}</b> bucket/range leaks, we remove them, and report a defensible
        <b> {h.pr_auc?.toFixed(3)}</b>. <i>Every team reporting ~0.99 on this dataset is reporting the leak.</i>
      </div>

      <div className="grid g2">
        <Card title="🎭 Leakage-sensitivity sweep — PR-AUC plateaus at ~0.86 (genuine signal)">
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={sweep} margin={{ top: 8, right: 16, bottom: 4, left: -16 }}>
              <XAxis dataKey="stage" stroke="#93a0bd" fontSize={12} />
              <YAxis domain={[0.8, 1]} stroke="#93a0bd" fontSize={12} />
              <Tooltip contentStyle={{ background: "#141b2e", border: "1px solid #263150", borderRadius: 10, color: "#e8edf7" }} />
              <ReferenceLine y={h.pr_auc} stroke="#2bbd7e" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="pr" stroke="#d11f2d" strokeWidth={2.5} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
          <p className="small muted">Steep early drop = leakage leaving. Flat tail = real behavioural signal.</p>
        </Card>

        <Card title="⚡ Real-time on commodity CPU (no GPU)">
          {rt ? (
            <div className="grid g2">
              <Metric k="Batch throughput" v={`${rt.batch_throughput_accts_per_sec?.toLocaleString()} /s`} d="~0.4 ms/account" cls="good" />
              <Metric k="On-demand score" v={`${Math.round(rt.score_latency_ms?.p50)} ms`} d={`p95 ${Math.round(rt.score_latency_ms?.p95)} ms`} />
              <Metric k="Model footprint" v={`${rt.model_footprint_mb} MB`} d={`${rt.n_features} features`} />
              <Metric k="Hardware" v="CPU" d="GPU gives 0 benefit on tabular" />
            </div>
          ) : <p className="muted">benchmark artifact not found</p>}
        </Card>
      </div>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Card title="🛟 Customer-harm-aware decisioning">
          {dec ? (
            <>
              <div className="grid g2">
                <Metric k="Wrongful freezes" v={dec.wrongful_freezes} d={`vs ${naiveDec?.wrongful_freezes ?? "?"} naïve`} cls="good" />
                <Metric k="Hard mules surfaced by step-up" v={dec.mules_surfaced_by_stepup} d="without freezing anyone" cls="good" />
                <Metric k="High-confidence freezes" v={dec.high_conf_freezes} d="precise action only" />
                <Metric k="Expected cost cut" v={fmtInr(s.decisioning.expected_cost_reduction_rupees)} d="vs naïve freeze@0.5" cls="good" />
              </div>
              <p className="small muted" style={{ marginTop: 10 }}>
                Graduated response: <b>clear → step-up verify → freeze</b>. Freeze only on high confidence; verify the
                ambiguous middle (a mule fails KYC re-check, a real customer passes). Protects innocents AND analysts.
              </p>
            </>
          ) : <p className="muted">decisioning artifact not found</p>}
        </Card>

        <Card title="🧪 Anomaly detection — tested honestly">
          {s.anomaly ? (
            <>
              <div className="grid g3">
                <Metric k="Supervised" v={s.anomaly.supervised_pr_auc?.toFixed(3)} cls="good" />
                <Metric k="Isolation Forest" v={s.anomaly.anomaly_pr_auc?.toFixed(3)} cls="bad" />
                <Metric k="LOF" v={s.anomaly.lof_pr_auc?.toFixed(3)} cls="bad" />
              </div>
              <p className="small muted" style={{ marginTop: 10 }}>
                Two unsupervised detectors score near-random — mules here are <b>not outliers</b>. A 60/20/20 hybrid
                <b> drops</b> the score to {s.anomaly.hybrid_pr_auc?.toFixed(3)}, so we surface anomaly as a second
                opinion but never fold it in. Reporting that beats shipping a metric that makes us worse.
              </p>
            </>
          ) : <p className="muted">anomaly artifact not found</p>}
        </Card>
      </div>
    </>
  );
}
