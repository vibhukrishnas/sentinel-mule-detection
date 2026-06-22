import { useRef, useState } from "react";
import { api, TxnIngest, AlertIngest } from "../api";
import { Metric, Card, fmtInr } from "../ui";

interface CrossChannelData {
  n_accounts: number; n_cross_channel: number; channels_seen: string[];
  accounts: { account: string; channels: string; n_channels: number; txns: number; total_value: number; cross_channel_flag: boolean }[];
}
interface StreamResp { n_total: number; n_flagged: number; events: { account: string; amount: number; suspicion: number; flag: boolean; reasons: string; action: string }[] }
interface RegResp { n_alerts: number; n_corroborated: number; by_source: Record<string, number>; alerts: { account: number | null; source: string; severity: string; model_risk: number | null; status: string }[]; note?: string }

const fetchGet = <T,>(path: string) => fetch(path).then(r => { if (!r.ok) throw new Error(`${path} -> ${r.status}`); return r.json() as Promise<T>; });
const fetchPost = <T,>(path: string, file: File) => { const fd = new FormData(); fd.append("file", file); return fetch(path, { method: "POST", body: fd }).then(r => { if (!r.ok) return r.json().then(d => { throw new Error(d.detail ?? `${path} -> ${r.status}`); }); return r.json() as Promise<T>; }); };

export default function Feeds() {
  const [mode, setMode] = useState<"txn" | "cc" | "reg" | "alert">("txn");
  const [txn, setTxn] = useState<TxnIngest | null>(null);
  const [stream, setStream] = useState<StreamResp | null>(null);
  const [cc, setCc] = useState<CrossChannelData | null>(null);
  const [reg, setReg] = useState<RegResp | null>(null);
  const [alerts, setAlerts] = useState<AlertIngest | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const txnRef = useRef<HTMLInputElement>(null);
  const ccRef = useRef<HTMLInputElement>(null);
  const regRef = useRef<HTMLInputElement>(null);
  const alRef = useRef<HTMLInputElement>(null);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setErr(null);
    try { await fn(); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  return (
    <>
      <h1 className="h1">🔌 Feeds & Transactions</h1>
      <p className="sub">
        Real-time ingestion of financial transactions, cross-channel bank data (UPI/IMPS/card/NEFT),
        govt regulatory feeds (I4C/NCRP/RBI), and fraud/TMS alert tickets. Scores suspicious transactions
        as they arrive, correlates accounts across payment rails, and corroborates every external alert
        against the deployed model risk.
      </p>

      <div className="row" style={{ gap: 6, marginBottom: 16, flexWrap: "wrap" }}>
        {(["txn","cc","reg","alert"] as const).map((id) => (
          <button key={id} className={mode === id ? "primary" : "ghost"} onClick={() => setMode(id)}>
            {{ txn: "💳 Transaction feed", cc: "🔗 Cross-channel", reg: "🏛️ Regulatory feed", alert: "📨 Alert / TMS feed" }[id]}
          </button>
        ))}
      </div>
      {err && <div className="banner err">{err}</div>}

      {/* ── TRANSACTION FEED ── */}
      {mode === "txn" && (
        <>
          <Card title="Ingest transaction feed — real-time suspicious-transaction detection">
            <p className="small muted">Any bank export with an amount column. Each transaction scored with TMS-style typologies (structuring, account-drain, pass-through, layering, velocity, cross-channel).</p>
            <div className="row" style={{ gap: 10, marginTop: 8 }}>
              <input ref={txnRef} type="file" accept=".csv" style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && run(async () => {
                  const r = await api.ingestTxns(e.target.files![0]); setTxn(r);
                  const sr = await fetchGet<StreamResp>("/ingest/stream/sample"); setStream(sr);
                })} />
              <button className="primary" onClick={() => txnRef.current?.click()}>⬆ Upload transaction CSV</button>
              <button className="ghost" onClick={() => run(async () => {
                const [r, sr] = await Promise.all([api.ingestSample(), fetchGet<StreamResp>("/ingest/stream/sample")]);
                setTxn(r); setStream(sr);
              })}>▶ Load synthetic sample</button>
              {busy && <span className="muted small">ingesting…</span>}
            </div>
          </Card>

          {txn && (
            <>
              <div className="grid g3" style={{ marginTop: 16 }}>
                <Metric k="Transactions ingested" v={txn.n_transactions.toLocaleString()} />
                <Metric k="🚩 Suspicious flagged" v={txn.n_flagged.toLocaleString()} d="suspicion ≥ 50" cls="bad" />
                <Metric k="Accounts implicated" v={txn.n_accounts_implicated.toLocaleString()} />
              </div>
              <Card title="Flagged transactions — reason-tagged">
                <div style={{ overflowX: "auto" }}>
                  <table><thead><tr><th>Account</th><th>Amount</th><th>Suspicion</th><th>Reasons</th></tr></thead>
                    <tbody>{txn.flagged.map((t, i) => (
                      <tr key={i}><td>{t.account ?? "—"}</td><td>{fmtInr(t.amount)}</td>
                        <td><b>{t.suspicion_score}</b></td><td className="small">{t.reasons}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              </Card>
              {txn.rollup.length > 0 && (
                <Card title="Account-level circulation view">
                  <div style={{ overflowX: "auto" }}>
                    <table><thead><tr><th>Account</th><th>Txns</th><th>Max suspicion</th><th>Suspicious txns</th><th>Suspicious value</th></tr></thead>
                      <tbody>{txn.rollup.map((r, i) => (
                        <tr key={i}><td>{r.account}</td><td>{r.txns}</td><td>{r.max_suspicion}</td>
                          <td>{r.suspicious_txns}</td><td>{fmtInr(r.suspicious_value)}</td></tr>
                      ))}</tbody>
                    </table>
                  </div>
                </Card>
              )}
            </>
          )}

          {stream && (
            <Card title="⚡ Real-time stream — transactions scored as they arrive">
              <p className="small muted">Each event gets a suspicion score and containment action the moment it lands. Same code path as a live stream ingest.</p>
              <div style={{ overflowX: "auto" }}>
                <table><thead><tr><th>Account</th><th>Amount</th><th>Suspicion</th><th>Reasons</th><th>Action</th></tr></thead>
                  <tbody>{stream.events.map((e, i) => (
                    <tr key={i}><td>{e.account}</td><td>{fmtInr(e.amount)}</td>
                      <td><b style={{ color: e.suspicion >= 70 ? "#e0414f" : "#e8a33d" }}>{e.suspicion}</b></td>
                      <td className="small">{e.reasons}</td>
                      <td className="small" style={{ color: e.action.includes("FREEZE") ? "#e0414f" : e.action.includes("HOLD") ? "#e8a33d" : "#93a0bd" }}>{e.action}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}

      {/* ── CROSS-CHANNEL ── */}
      {mode === "cc" && (
        <>
          <Card title="Cross-channel bank data — UPI · IMPS · card · NEFT correlation">
            <p className="small muted">Merges transactions across all payment rails. Accounts active on ≥2 rails = cross-channel layering signal that single-channel TMS cannot see.</p>
            <div className="row" style={{ gap: 10, marginTop: 8 }}>
              <input ref={ccRef} type="file" accept=".csv" style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && run(async () => {
                  const r = await fetchPost<{ cross_channel: CrossChannelData }>("/ingest/crosschannel", e.target.files![0]);
                  setCc(r.cross_channel);
                })} />
              <button className="primary" onClick={() => ccRef.current?.click()}>⬆ Upload multi-rail CSV</button>
              <button className="ghost" onClick={() => run(async () => {
                const r = await fetchGet<{ cross_channel: CrossChannelData }>("/ingest/crosschannel/sample");
                setCc(r.cross_channel);
              })}>▶ Load synthetic multi-rail feed</button>
              {busy && <span className="muted small">analysing…</span>}
            </div>
          </Card>

          {cc && (
            <>
              <div className="grid g3" style={{ marginTop: 16 }}>
                <Metric k="Payment rails" v={cc.channels_seen.join(", ")} />
                <Metric k="Accounts analysed" v={cc.n_accounts} />
                <Metric k="🔗 Cross-channel" v={cc.n_cross_channel} d="active on ≥2 rails — layering risk" cls="bad" />
              </div>
              <Card title="Cross-channel account view (multi-rail first)">
                <div style={{ overflowX: "auto" }}>
                  <table><thead><tr><th>Account</th><th>Channels</th><th># Rails</th><th>Txns</th><th>Total value</th><th>Flag</th></tr></thead>
                    <tbody>{cc.accounts.map((a, i) => (
                      <tr key={i} style={{ background: a.cross_channel_flag ? "rgba(224,65,79,.07)" : undefined }}>
                        <td>{a.account}</td><td className="small">{a.channels}</td><td>{a.n_channels}</td>
                        <td>{a.txns}</td><td>{fmtInr(a.total_value)}</td>
                        <td style={{ color: a.cross_channel_flag ? "#e0414f" : "#93a0bd" }}>{a.cross_channel_flag ? "🔗 multi-rail" : "—"}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              </Card>
            </>
          )}
        </>
      )}

      {/* ── REGULATORY FEED ── */}
      {mode === "reg" && (
        <>
          <Card title="Regulatory feed — I4C / NCRP / RBI cyber-fraud tickets">
            <p className="small muted">Normalises govt regulatory cyber-fraud tickets through the I4C/NCRP connector then corroborates each against the deployed model. Account flagged by both → immediate escalation.</p>
            <div className="row" style={{ gap: 10, marginTop: 8 }}>
              <input ref={regRef} type="file" accept=".csv" style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && run(async () => {
                  const r = await fetchPost<RegResp>("/ingest/regulatory", e.target.files![0]); setReg(r);
                })} />
              <button className="primary" onClick={() => regRef.current?.click()}>⬆ Upload regulatory CSV</button>
              <button className="ghost" onClick={() => run(async () => {
                const r = await fetchGet<RegResp>("/ingest/regulatory/sample"); setReg(r);
              })}>▶ Load synthetic I4C/NCRP tickets</button>
              {busy && <span className="muted small">ingesting…</span>}
            </div>
          </Card>

          {reg && (
            <>
              <div className="grid g3" style={{ marginTop: 16 }}>
                <Metric k="Regulatory tickets" v={reg.n_alerts} />
                <Metric k="✅ Corroborated by model" v={reg.n_corroborated} d="risk ≥70 — escalate" cls="good" />
                <Metric k="Source" v="Govt-I4C" />
              </div>
              <Card title="Corroboration — ticket × model risk">
                <div style={{ overflowX: "auto" }}>
                  <table><thead><tr><th>Account</th><th>Source</th><th>Severity</th><th>Model risk</th><th>Status</th></tr></thead>
                    <tbody>{reg.alerts.map((a, i) => (
                      <tr key={i}><td>{a.account ?? "—"}</td><td>{a.source}</td><td>{a.severity}</td>
                        <td>{a.model_risk ?? "—"}</td>
                        <td className="small" style={{ color: a.status.startsWith("CORROBORATED") ? "#2bbd7e" : undefined }}>{a.status}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
                {reg.note && <p className="small muted" style={{ marginTop: 8 }}>{reg.note}</p>}
              </Card>
            </>
          )}
        </>
      )}

      {/* ── ALERT / TMS FEED ── */}
      {mode === "alert" && (
        <>
          <Card title="Alert / TMS feed — fraud-monitoring & TMS corroboration">
            <p className="small muted">Ingest external TMS / fraud-monitoring alert tickets and corroborate against the deployed model risk.</p>
            <div className="row" style={{ gap: 10, marginTop: 8 }}>
              <input ref={alRef} type="file" accept=".csv" style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && run(async () => {
                  const r = await api.ingestAlerts(e.target.files![0]); setAlerts(r);
                })} />
              <button className="primary" onClick={() => alRef.current?.click()}>⬆ Upload alert-ticket CSV</button>
              {busy && <span className="muted small">ingesting…</span>}
            </div>
          </Card>

          {alerts && (
            <>
              <div className="grid g3" style={{ marginTop: 16 }}>
                <Metric k="Alerts ingested" v={alerts.n_alerts} />
                <Metric k="✅ Corroborated" v={alerts.n_corroborated} d="model risk ≥ 70" cls="good" />
                <Metric k="Sources" v={Object.keys(alerts.by_source).join(", ")} />
              </div>
              <Card title="Corroboration">
                <div style={{ overflowX: "auto" }}>
                  <table><thead><tr><th>Account</th><th>Source</th><th>Severity</th><th>Model risk</th><th>Status</th></tr></thead>
                    <tbody>{alerts.alerts.map((a, i) => (
                      <tr key={i}><td>{a.account ?? "—"}</td><td>{a.source}</td><td>{a.severity}</td>
                        <td>{a.model_risk ?? "—"}</td><td className="small">{a.status}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              </Card>
            </>
          )}
        </>
      )}
    </>
  );
}
