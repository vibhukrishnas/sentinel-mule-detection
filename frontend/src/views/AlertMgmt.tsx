import { useEffect, useMemo, useState } from "react";
import { api, AccountRow, Summary } from "../api";
import { Metric, Card, Badge, Loading } from "../ui";

type Status = "NEW" | "INVESTIGATING" | "ESCALATED" | "CLEARED";
const NEXT: Record<Status, Status[]> = {
  NEW: ["INVESTIGATING", "ESCALATED", "CLEARED"],
  INVESTIGATING: ["ESCALATED", "CLEARED"],
  ESCALATED: ["CLEARED"],
  CLEARED: ["NEW"],
};
const statusColor: Record<Status, string> = { NEW: "var(--amber)", INVESTIGATING: "var(--navy)", ESCALATED: "var(--red)", CLEARED: "var(--green)" };

export default function AlertMgmt({ s }: { s: Summary }) {
  const [rows, setRows] = useState<AccountRow[]>([]);
  const [thr, setThr] = useState(0.5);
  const [status, setStatus] = useState<Record<number, Status>>({});
  const [owner, setOwner] = useState<Record<number, string>>({});
  useEffect(() => { api.accounts().then((d) => setRows(d.accounts)); }, []);

  // alerts = whatever is loaded (sample OR uploaded) scoring >= threshold
  const alerts = useMemo(() => rows.filter((r) => r.probability >= thr).sort((a, b) => b.risk_score - a.risk_score), [rows, thr]);
  const stOf = (id: number): Status => status[id] ?? "NEW";
  const set = (id: number, st: Status) => setStatus((m) => ({ ...m, [id]: st }));
  const counts = useMemo(() => {
    const c: Record<string, number> = { NEW: 0, INVESTIGATING: 0, ESCALATED: 0, CLEARED: 0 };
    alerts.forEach((a) => c[stOf(a.account_id)]++); return c;
  }, [alerts, status]);

  if (!rows.length) return <Loading what="alerts" />;
  const handled = counts.INVESTIGATING + counts.ESCALATED + counts.CLEARED;

  return (
    <>
      <h1 className="h1">Alert Management</h1>
      <p className="sub">Every account scoring at/above the alert threshold becomes a case with a status and an owner — full accountability, on the bank sample <i>or</i> any uploaded dataset.</p>

      <Card>
        <div className="row" style={{ alignItems: "center", gap: 14 }}>
          <div className="thr" style={{ flex: 1 }}>
            <span className="small muted">Alert threshold</span>
            <input type="range" min={0.05} max={0.99} step={0.01} value={thr} onChange={(e) => setThr(Number(e.target.value))} style={{ flex: 1 }} />
            <b>{thr.toFixed(2)}</b>
          </div>
          <span className="small muted">Lower = catch more (more workload); higher = fewer, higher-confidence alerts.</span>
        </div>
      </Card>

      <div className="grid g4" style={{ marginTop: 16 }}>
        <Metric k="🚨 Open alerts" v={alerts.length} d={`@ threshold ${thr.toFixed(2)}`} cls="accent" />
        <Metric k="🟡 New / untouched" v={counts.NEW} />
        <Metric k="🔵 Investigating + 🔴 Escalated" v={counts.INVESTIGATING + counts.ESCALATED} />
        <Metric k="✅ Cleared" v={counts.CLEARED} cls="good" />
      </div>
      <div className="banner ok" style={{ marginTop: 12 }}>
        Handled <b>{handled}/{alerts.length}</b> alerts this session. SLA suggestion: Critical (≥90) same-day, High (70–89) 48h.
        Status &amp; owner are tracked per account so the desk has an auditable trail.
      </div>

      <Card title="Alert queue (act on each — status & owner are logged)">
        <table>
          <thead><tr><th>Account</th><th>Risk</th><th>Band</th><th>Prob</th><th>Truth</th><th>Owner</th><th>Status</th><th>Action</th></tr></thead>
          <tbody>
            {alerts.slice(0, 60).map((a) => {
              const st = stOf(a.account_id);
              return (
                <tr key={a.account_id}>
                  <td>#{a.account_id}</td>
                  <td><b>{a.risk_score}</b></td>
                  <td><Badge band={a.band} /></td>
                  <td className="muted">{(a.probability * 100).toFixed(1)}%</td>
                  <td className={a.ground_truth === "MULE" ? "bad" : "muted"}>{a.ground_truth ?? "—"}</td>
                  <td>
                    <input style={{ width: 90, padding: "5px 7px" }} placeholder="assign…" value={owner[a.account_id] ?? ""}
                      onChange={(e) => setOwner((m) => ({ ...m, [a.account_id]: e.target.value }))} />
                  </td>
                  <td><span style={{ color: statusColor[st], fontWeight: 700, fontSize: 12 }}>{st}</span></td>
                  <td>
                    <div className="row" style={{ gap: 4 }}>
                      {NEXT[st].map((n) => (
                        <button key={n} className="ghost" style={{ padding: "4px 8px", fontSize: 11 }} onClick={() => set(a.account_id, n)}>{n === "INVESTIGATING" ? "Investigate" : n.charAt(0) + n.slice(1).toLowerCase()}</button>
                      ))}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {alerts.length > 60 && <p className="small muted">Showing top 60 of {alerts.length} alerts by risk.</p>}
      </Card>
    </>
  );
}
