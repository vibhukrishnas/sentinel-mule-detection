import { useEffect, useState } from "react";
import { api, RingsResp, NetResp, AccountRow } from "../api";
import { Card, Metric, Loading, fmtInr } from "../ui";

const W = 540, H = 420, CX = W / 2, CY = H / 2, R = 150;
const col = (risk: number) => (risk >= 90 ? "#e0414f" : risk >= 70 ? "#e8a33d" : risk >= 40 ? "#4f7cff" : "#2bbd7e");

function circle(items: number[], r = R) {
  const n = Math.max(items.length, 1); const m: Record<number, [number, number]> = {};
  items.forEach((it, i) => { const a = (2 * Math.PI * i) / n - Math.PI / 2; m[it] = [CX + r * Math.cos(a), CY + r * Math.sin(a)]; });
  return m;
}

const Legend = ({ showCenter }: { showCenter?: boolean }) => (
  <div className="row" style={{ gap: 16, flexWrap: "wrap", margin: "6px 0 10px", fontSize: 12.5 }}>
    {showCenter && <span><span className="dot" style={{ background: "#fff", border: "3px solid #7b1fa2" }} /> selected account</span>}
    <span><span className="dot" style={{ background: "#e0414f" }} /> Critical (risk ≥ 90)</span>
    <span><span className="dot" style={{ background: "#e8a33d" }} /> High (70–89)</span>
    <span><span className="dot" style={{ background: "#4f7cff" }} /> Medium (40–69)</span>
    <span><span className="dot" style={{ background: "#2bbd7e" }} /> Low (&lt; 40)</span>
    <span className="muted">· node size ∝ risk</span>
  </div>
);

function AccountMap({ rows }: { rows: AccountRow[] }) {
  const [sel, setSel] = useState<number | null>(rows[0]?.account_id ?? null);
  const [net, setNet] = useState<NetResp | null>(null);
  const [busy, setBusy] = useState(false);
  const load = (id: number) => { setBusy(true); api.net(id).then(setNet).finally(() => setBusy(false)); };
  useEffect(() => { if (sel != null) load(sel); }, []);
  const peers = net ? net.nodes.filter((n) => !n.center).map((n) => n.id) : [];
  const pos = circle(peers);
  if (net) { const c = net.nodes.find((n) => n.center); if (c) pos[c.id] = [CX, CY]; }
  return (
    <>
      <div className="row" style={{ alignItems: "center", marginBottom: 8 }}>
        <select value={sel ?? ""} onChange={(e) => { const id = Number(e.target.value); setSel(id); load(id); }}>
          {rows.map((r) => <option key={r.account_id} value={r.account_id}>Account #{r.account_id} · risk {r.risk_score}</option>)}
        </select>
        {busy && <span className="muted small">building map…</span>}
      </div>
      <Legend showCenter />
      {net && (
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: "#0a0f1d", borderRadius: 12 }}>
          <defs><marker id="ar" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,3 L0,6 Z" fill="#d11f2d" /></marker></defs>
          {net.edges.map((e, i) => { const a = pos[e.source], b = pos[e.target]; if (!a || !b) return null;
            return <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#d11f2d" strokeOpacity={0.5} strokeWidth={1 + e.weight * 2} markerEnd="url(#ar)" />; })}
          {net.nodes.map((n) => { const p = pos[n.id]; if (!p) return null;
            return (<g key={n.id}>
              <circle cx={p[0]} cy={p[1]} r={n.center ? 18 : 9 + n.risk / 12} fill={col(n.risk)} stroke={n.center ? "#7b1fa2" : "#0a0f1d"} strokeWidth={n.center ? 4 : 1} />
              <text x={p[0]} y={p[1] - (n.center ? 24 : 16)} fill="#cdd7ef" fontSize={n.center ? 11 : 9} textAnchor="middle" fontWeight={n.center ? 700 : 400}>#{n.id}</text>
            </g>); })}
        </svg>
      )}
      <p className="small muted">Centre = the selected account; spokes = its closest behavioural look-alikes; arrows funnel look-alikes → the higher-risk node. <b>Works for every account, including uploaded data.</b></p>
    </>
  );
}

function RingGraph({ data }: { data: RingsResp }) {
  const ringsMeta = data.rings?.rings ?? [];
  const [sel, setSel] = useState(0);
  if (!data.network || !ringsMeta.length) return <p className="muted">Ring view available on the bank sample (upload a labelled set to recompute).</p>;
  const ring = ringsMeta[sel]; const members: number[] = ring.members; const mset = new Set(members);
  const pos = circle(members); const riskMap: Record<number, number> = {};
  data.network.nodes.forEach((n) => (riskMap[n.id] = n.risk));
  const edges = data.network.edges.filter((e) => mset.has(e.source) && mset.has(e.target));
  return (
    <>
      <select value={sel} onChange={(e) => setSel(Number(e.target.value))} style={{ marginBottom: 8 }}>
        {ringsMeta.map((r: any, i: number) => <option key={i} value={i}>Ring #{r.ring_id} · {r.size} accounts · {fmtInr(r.exposure_rupees)}</option>)}
      </select>
      <Legend />
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: "#0a0f1d", borderRadius: 12 }}>
        {edges.map((e, i) => { const a = pos[e.source], b = pos[e.target]; return <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#2a3358" />; })}
        {members.map((m) => { const p = pos[m]; const rk = riskMap[m] ?? 0; return (
          <g key={m}><circle cx={p[0]} cy={p[1]} r={m === ring.rep_account ? 14 : 8} fill={col(rk)} stroke="#fff" strokeWidth={1} />
          <text x={p[0]} y={p[1] - 13} fill="#93a0bd" fontSize={9} textAnchor="middle">#{m}</text></g>); })}
      </svg>
      <p className="small muted">Validated candidate ring — ~30× tighter than a random group, stable under feature subsampling (Jaccard 0.82). Large node = ring lead.</p>
    </>
  );
}

export default function Network() {
  const [rings, setRings] = useState<RingsResp | null>(null);
  const [rows, setRows] = useState<AccountRow[] | null>(null);
  useEffect(() => { api.rings().then(setRings); api.accounts().then((d) => setRows(d.accounts)); }, []);
  if (!rings || !rows) return <Loading what="network maps" />;
  const rv = rings.rings;
  return (
    <>
      <h1 className="h1">Account Network</h1>
      <p className="sub">The network behind an account — its look-alike map (any account, incl. uploaded) and the validated candidate rings.</p>
      {rv && (
        <div className="grid g3">
          <Metric k="Candidate rings" v={rv.n_candidate_rings} />
          <Metric k="Mules grouped" v={`${rv.mules_in_rings}/${rv.n_mules}`} />
          <Metric k="Ring #1 stability" v={(rv.validation?.ring1_subsample_stability_jaccard ?? 0).toFixed(2)} d="Jaccard under subsampling" />
        </div>
      )}
      <div className="card" style={{ marginTop: 16 }}>
        <h3>🗺️ Account network map — live, any account (incl. uploaded)</h3>
        <AccountMap rows={rows} />
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h3>🕸️ Validated candidate rings</h3>
        <RingGraph data={rings} />
      </div>
    </>
  );
}
