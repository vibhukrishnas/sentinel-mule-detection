import { useEffect, useState } from "react";
import { api, AccountRow, AccountDetail } from "../api";
import { Metric, Card, Badge, Bar, Loading } from "../ui";

export default function Investigate({ threshold }: { threshold: number }) {
  const [rows, setRows] = useState<AccountRow[]>([]);
  const [pick, setPick] = useState<number | null>(null);
  const [detail, setDetail] = useState<AccountDetail | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { api.accounts().then((d) => { setRows(d.accounts); }); }, []);

  const check = (id: number) => {
    setBusy(true); setDetail(null);
    api.account(id).then(setDetail).finally(() => setBusy(false));
  };

  const random = () => {
    if (!rows.length) return;
    const r = rows[Math.floor(Math.random() * rows.length)];
    setPick(r.account_id); check(r.account_id);
  };

  return (
    <>
      <h1 className="h1">Investigate an account</h1>
      <p className="sub">Pick an account → the model scores it live (LightGBM + SHAP via the API), and explains why.</p>

      <Card>
        <div className="row" style={{ alignItems: "center" }}>
          <select value={pick ?? ""} onChange={(e) => setPick(Number(e.target.value))} style={{ minWidth: 220 }}>
            <option value="" disabled>Select account…</option>
            {rows.map((r) => <option key={r.account_id} value={r.account_id}>Account #{r.account_id}</option>)}
          </select>
          <button className="primary" onClick={() => pick != null && check(pick)}>🔍 Check risk score</button>
          <button className="primary" style={{ background: "#1b2440", color: "#e8edf7" }} onClick={random}>🎲 Random</button>
        </div>
      </Card>

      {busy && <Loading what="model score" />}

      {detail && (
        <div className="grid g2" style={{ marginTop: 16 }}>
          <Card title={`Account #${detail.account_id}`}>
            <div className="row" style={{ alignItems: "baseline", gap: 12 }}>
              <div className="v" style={{ fontSize: 44, fontWeight: 800 }}>{detail.risk_score}<span className="muted" style={{ fontSize: 18 }}>/100</span></div>
              <Badge band={detail.band} />
            </div>
            <div style={{ margin: "12px 0" }}><Bar pct={detail.risk_score} /></div>
            <div className="grid g2">
              <Metric k="Mule probability" v={`${(detail.probability * 100).toFixed(1)}%`} />
              <Metric k="Confidence tier" v={detail.confidence_tier.replace("-", " ")}
                cls={detail.confidence_tier === "UNCERTAIN" ? "warn" : "muted"} />
              <Metric k="🔒 Model trust (this call)" v={`${detail.data_quality.model_trust}%`} />
              <Metric k="📋 Data completeness" v={`${(detail.data_quality.completeness * 100).toFixed(0)}%`}
                d={`${detail.data_quality.n_blank} blank features`} />
            </div>
            {detail.ground_truth && (
              <p className="small" style={{ marginTop: 10 }}>
                Ground truth: <b className={detail.ground_truth === "MULE" ? "bad" : "good"}>{detail.ground_truth.toUpperCase()}</b>
                {" "}{(detail.ground_truth === "MULE") === (detail.probability >= threshold) ? "✅ model agrees" : `⚠️ disagrees at ${threshold.toFixed(2)}`}
              </p>
            )}
            {detail.confidence_tier === "UNCERTAIN" && (
              <div className="banner warn" style={{ marginTop: 10 }}>🤔 <b>UNCERTAIN — route to analyst.</b> The model declines to auto-decide; step-up verification recommended over a freeze.</div>
            )}
          </Card>

          <Card title="Why — top risk drivers (SHAP)">
            <table>
              <thead><tr><th>Factor</th><th>Value</th><th>Effect</th><th>Impact</th></tr></thead>
              <tbody>
                {detail.top_drivers.map((d, i) => (
                  <tr key={i}>
                    <td>{d.label ?? d.feature}</td>
                    <td className="muted">{String(d.value_readable ?? d.value)}</td>
                    <td className={d.shap > 0 ? "bad" : "good"}>{d.shap > 0 ? "▲ raises" : "▼ lowers"}</td>
                    <td>{Math.abs(d.shap).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ gridColumn: "1 / -1" }}>
            <Card title="📄 Investigation report (SAR / adverse-action ready)">
              <div className="report">{detail.report}</div>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
