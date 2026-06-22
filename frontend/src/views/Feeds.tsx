import { useRef, useState } from "react";
import { api, TxnIngest, AlertIngest } from "../api";
import { Metric, Card, fmtInr } from "../ui";

// Phase-2 ingestion: upload a transaction feed -> suspicious-transaction detection;
// upload an alert-ticket feed -> corroboration against the deployed model's account risk.
export default function Feeds() {
  const [mode, setMode] = useState<"txn" | "alert">("txn");
  const [txn, setTxn] = useState<TxnIngest | null>(null);
  const [alerts, setAlerts] = useState<AlertIngest | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const txnRef = useRef<HTMLInputElement>(null);
  const alRef = useRef<HTMLInputElement>(null);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setErr(null);
    try { await fn(); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  return (
    <>
      <h1 className="h1">🔌 Feeds & Transactions</h1>
      <p className="sub">
        PS2 asks the system to ingest financial transactions and fraud/TMS/govt alert feeds and prevent circulation.
        BOI data is an account snapshot, so this is the working ingestion engine — rule-based, explainable, runs on
        any feed a bank plugs in. No training, no fabricated data; every flag carries its reasons.
      </p>

      <div className="row" style={{ gap: 8, marginBottom: 14 }}>
        <button className={mode === "txn" ? "primary" : "ghost"} onClick={() => setMode("txn")}>💳 Transaction feed</button>
        <button className={mode === "alert" ? "primary" : "ghost"} onClick={() => setMode("alert")}>📨 Alert / ticket feed</button>
      </div>
      {err && <div className="banner err">{err}</div>}

      {mode === "txn" && (
        <>
          <Card title="Ingest a transaction feed">
            <p className="small muted">PaySim-style or any bank export with an amount column (amount / amt / txn_amount).
              Type, balances and sender/receiver are used if present.</p>
            <div className="row" style={{ gap: 10, marginTop: 8 }}>
              <input ref={txnRef} type="file" accept=".csv" style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && run(async () => setTxn(await api.ingestTxns(e.target.files![0])))} />
              <button className="primary" onClick={() => txnRef.current?.click()}>⬆ Upload transaction CSV</button>
              <button className="ghost" onClick={() => run(async () => setTxn(await api.ingestSample()))}>▶ Load synthetic sample</button>
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
              <Card title="Flagged transactions — highest suspicion first">
                <div style={{ overflowX: "auto" }}>
                  <table>
                    <thead><tr><th>Account</th><th>Amount</th><th>Suspicion</th><th>Reasons</th></tr></thead>
                    <tbody>
                      {txn.flagged.map((t, i) => (
                        <tr key={i}>
                          <td>{t.account ?? "—"}</td><td>{fmtInr(t.amount)}</td>
                          <td><b>{t.suspicion_score}</b></td><td className="small">{t.reasons}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
              {txn.rollup.length > 0 && (
                <Card title="Account-level circulation view — where to act to stop the money moving">
                  <div style={{ overflowX: "auto" }}>
                    <table>
                      <thead><tr><th>Account</th><th>Txns</th><th>Max suspicion</th><th>Suspicious txns</th><th>Suspicious value</th></tr></thead>
                      <tbody>
                        {txn.rollup.map((r, i) => (
                          <tr key={i}><td>{r.account}</td><td>{r.txns}</td><td>{r.max_suspicion}</td>
                            <td>{r.suspicious_txns}</td><td>{fmtInr(r.suspicious_value)}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}
              <p className="small muted" style={{ marginTop: 10 }}>Typologies scored: high-value spikes, structuring
                (just-under-threshold smurfing), account-drain, pass-through, cash-out/transfer channels,
                transfer→cash-out layering chains, and high velocity.</p>
            </>
          )}
        </>
      )}

      {mode === "alert" && (
        <>
          <Card title="Ingest external alert tickets (TMS / fraud-monitoring / govt cyber-fraud)">
            <p className="small muted">Corroborate each ticket against the model's account risk. An account flagged by
              both the feed and the model is the highest-priority, most-defensible escalation. CSV needs an account
              column (account / account_id / acct); optional source & severity.</p>
            <div className="row" style={{ gap: 10, marginTop: 8 }}>
              <input ref={alRef} type="file" accept=".csv" style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && run(async () => setAlerts(await api.ingestAlerts(e.target.files![0])))} />
              <button className="primary" onClick={() => alRef.current?.click()}>⬆ Upload alert-ticket CSV</button>
              {busy && <span className="muted small">ingesting…</span>}
            </div>
          </Card>

          {alerts && (
            <>
              <div className="grid g3" style={{ marginTop: 16 }}>
                <Metric k="Alerts ingested" v={alerts.n_alerts} />
                <Metric k="✅ Corroborated by model" v={alerts.n_corroborated} d="model risk ≥ 70 — escalate" cls="good" />
                <Metric k="Feed sources" v={Object.keys(alerts.by_source).length} />
              </div>
              <Card title="Corroboration">
                <div style={{ overflowX: "auto" }}>
                  <table>
                    <thead><tr><th>Account</th><th>Source</th><th>Severity</th><th>Model risk</th><th>Status</th></tr></thead>
                    <tbody>
                      {alerts.alerts.map((a, i) => (
                        <tr key={i}><td>{a.account ?? "—"}</td><td>{a.source}</td><td>{a.severity}</td>
                          <td>{a.model_risk ?? "—"}</td><td className="small">{a.status}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
              <p className="small muted" style={{ marginTop: 10 }}>Cross-referencing two independent signals cuts false
                escalations and gives an auditable reason to act.</p>
            </>
          )}
        </>
      )}
    </>
  );
}
