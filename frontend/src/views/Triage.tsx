import { useEffect, useState } from "react";
import { api, AccountRow, Summary } from "../api";
import { Metric, Card, Badge, Loading } from "../ui";

const QueueTable = ({ rows }: { rows: AccountRow[] }) => (
  <table>
    <thead><tr><th>Account</th><th>Risk</th><th>Band</th><th>Prob</th><th>Tier</th><th>Truth</th></tr></thead>
    <tbody>
      {rows.slice(0, 15).map((r) => (
        <tr key={r.account_id}>
          <td>#{r.account_id}</td><td><b>{r.risk_score}</b></td><td><Badge band={r.band} /></td>
          <td className="muted">{(r.probability * 100).toFixed(1)}%</td>
          <td className="small muted">{r.confidence_tier.replace("-", " ")}</td>
          <td className={r.ground_truth === "MULE" ? "bad" : "muted"}>{r.ground_truth ?? "—"}</td>
        </tr>
      ))}
    </tbody>
  </table>
);

export default function Triage({ s, threshold }: { s: Summary; threshold: number }) {
  const [rows, setRows] = useState<AccountRow[]>([]);
  useEffect(() => { api.accounts().then((d) => setRows(d.accounts)); }, []);
  if (!rows.length) return <Loading what="queue" />;
  const sorted = [...rows].sort((a, b) => b.risk_score - a.risk_score);
  const crit = sorted.filter((r) => r.risk_score >= 90);
  const urg = sorted.filter((r) => r.risk_score >= 70 && r.risk_score < 90);
  const review = sorted.filter((r) => r.confidence_tier === "UNCERTAIN");
  const alerts = sorted.filter((r) => r.probability >= threshold);
  const ab = s.abstention;
  return (
    <>
      <h1 className="h1">Analyst triage queue</h1>
      <p className="sub">What a fraud desk works down each morning. The <b>alert threshold</b> (top bar) sets what's flagged — drag it and these move.</p>
      <div className="grid g4">
        <Metric k={`🚨 Alerts @ ${threshold.toFixed(2)}`} v={alerts.length} d="driven by the threshold dial" cls="accent" />
        <Metric k="🔴 Critical (≥90)" v={crit.length} cls="bad" />
        <Metric k="🟠 Urgent (70–89)" v={urg.length} cls="warn" />
        <Metric k="🤔 Needs review (abstain)" v={review.length} />
      </div>
      {ab && (
        <div className="banner ok" style={{ marginTop: 16 }}>
          On out-of-fold data the model <b>auto-decides {(ab.coverage_auto_decided * 100).toFixed(0)}%</b> of accounts
          at <b>{(ab.auto_zone_error_rate * 100).toFixed(2)}%</b> error and routes the ambiguous
          {" "}<b>{(ab.review_rate * 100).toFixed(1)}%</b> to a human — instead of overconfident calls on the hard tail.
        </div>
      )}
      <div className="grid g3" style={{ marginTop: 16 }}>
        <Card title="🔴 Critical — escalate now"><QueueTable rows={crit} /></Card>
        <Card title="🟠 Urgent — same-day review"><QueueTable rows={urg} /></Card>
        <Card title="🤔 Needs human review"><QueueTable rows={review} /></Card>
      </div>
    </>
  );
}
