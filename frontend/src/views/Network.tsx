import { useEffect, useState } from "react";
import { api, RingsResp, FlowResp, NetResp, AccountRow } from "../api";
import { Card, Metric, Loading, fmtInr, Badge } from "../ui";

const W = 540, H = 420, CX = W / 2, CY = H / 2, R = 150;
const col = (risk: number) => (risk >= 90 ? "#e0414f" : risk >= 70 ? "#e8a33d" : risk >= 40 ? "#4f7cff" : "#2bbd7e");

function circle(items: number[], r = R) {
  const n = Math.max(items.length, 1);
  const m: Record<number, [number, number]> = {};
  items.forEach((it, i) => { const a = (2 * Math.PI * i) / n - Math.PI / 2; m[it] = [CX + r * Math.cos(a), CY + r * Math.sin(a)]; });
  return m;
}

const Arrowhead = () => (
  <defs><marker id="ar" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L9,3 L0,6 Z" fill="#d11f2d" /></marker></defs>
);

/* Per-account ego-network — works for ANY account incl. uploaded data */
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
      <div className="row" style={{ alignItems: "center", marginBottom: 10 }}>
        <select value={sel ?? ""} onChange={(e) => { const id = Number(e.target.value); setSel(id); load(id); }}>
          {rows.map((r) => <option key={r.account_id} value={r.account_id}>Account #{r.account_id} · risk {r.risk_score}</option>)}
        </select>
        {busy && <span className="muted small">building map…</span>}
      </div>
      {net && (
        <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: "#0a0f1d", borderRadius: 12 }}>
          <Arrowhead />
          {net.edges.map((e, i) => {
            const a = pos[e.source], b = pos[e.target]; if (!a || !b) return null;
            return <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#d11f2d" strokeOpacity={0.55}
              strokeWidth={1 + e.weight * 2} markerEnd="url(#ar)" />;
          })}
          {net.nodes.map((n) => {
            const p = pos[n.id]; if (!p) return null;
            return (
              <g key={n.id}>
                <circle cx={p[0]} cy={p[1]} r={n.center ? 18 : 9 + n.risk / 12} fill={col(n.risk)}
                  stroke={n.center ? "#fff" : "#0a0f1d"} strokeWidth={n.center ? 3 : 1} />
                <text x={p[0]} y={p[1] - (n.center ? 24 : 16)} fill="#cdd7ef" fontSize={n.center ? 11 : 9} textAnchor="middle" fontWeight={n.center ? 700 : 400}>#{n.id}</text>
              </g>
            );
          })}
        </svg>
      )}
      <p className="small muted">Centre (white ring) = the selected account; spokes = its closest behavioural look-alikes; arrows funnel look-alikes → higher-risk node. Node colour/size ∝ risk. <b>Works for every account, including uploaded data.</b> Behavioural-similarity proxy (BOI is a snapshot with no transaction edges).</p>
    </>
  );
}

function RingGraph({ data }: { data: RingsResp }) {
  const ringsMeta = data.rings?.rings ?? [];
  const [sel, setSel] = useState(0);
  if (!data.network || !ringsMeta.length) return <p className="muted">ring data not found (run on the committed sample)</p>;
  const ring = ringsMeta[sel]; const members: number[] = ring.members; const mset = new Set(members);
  const pos = circle(members); const riskMap: Record<number, number> = {};
  data.network.nodes.forEach((n) => (riskMap[n.id] = n.risk));
  const edges = data.network.edges.filter((e) => mset.has(e.source) && mset.has(e.target));
  return (
    <>
      <select value={sel} onChange={(e) => setSel(Number(e.target.value))} style={{ marginBottom: 10 }}>
        {ringsMeta.map((r: any, i: number) => <option key={i} value={i}>Ring #{r.ring_id} · {r.size} accts · {fmtInr(r.exposure_rupees)}</option>)}
      </select>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: "#0a0f1d", borderRadius: 12 }}>
        {edges.map((e, i) => { const a = pos[e.source], b = pos[e.target]; return <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#2a3358" />; })}
        {members.map((m) => { const p = pos[m]; const rk = riskMap[m] ?? 0; return (
          <g key={m}><circle cx={p[0]} cy={p[1]} r={m === ring.rep_account ? 13 : 8} fill={col(rk)} stroke="#fff" strokeWidth={1} />
          <text x={p[0]} y={p[1] - 13} fill="#93a0bd" fontSize={9} textAnchor="middle">#{m}</text></g>); })}
      </svg>
      <p className="small muted">Validated candidate rings among the 81 known mules · ~30× tighter than chance, Jaccard 0.82 stable.</p>
    </>
  );
}

function FlowGraph({ data }: { data: FlowResp }) {
  const [sel, setSel] = useState(0);
  if (!data.typologies?.length) return <p className="muted">money-flow data not found</p>;
  const t = data.typologies[sel]; const deg: Record<number, number> = {};
  t.edges.forEach((e) => { deg[e.source] = (deg[e.source] ?? 0) + 1; deg[e.target] = (deg[e.target] ?? 0) + 1; });
  let pos: Record<number, [number, number]>;
  if (t.type.startsWith("fan")) { const hub = t.accounts.reduce((a, b) => (deg[a] >= deg[b] ? a : b)); pos = circle(t.accounts.filter((a) => a !== hub)); pos[hub] = [CX, CY]; }
  else pos = circle(t.accounts);
  return (
    <>
      <select value={sel} onChange={(e) => setSel(Number(e.target.value))} style={{ marginBottom: 10 }}>
        {data.typologies.map((tp, i) => <option key={i} value={i}>{tp.type} · {tp.n_accounts} accts · {fmtInr(tp.total_amount)} moved</option>)}
      </select>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: "#0a0f1d", borderRadius: 12 }}>
        <Arrowhead />
        {t.edges.map((e, i) => { const a = pos[e.source], b = pos[e.target]; const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2; return (
          <g key={i}><line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#d11f2d" strokeWidth={2} markerEnd="url(#ar)" />
          <text x={mx} y={my - 4} fill="#c8d3ec" fontSize={9} textAnchor="middle">{fmtInr(e.amount)}</text></g>); })}
        {t.accounts.map((a) => { const p = pos[a]; return (
          <g key={a}><circle cx={p[0]} cy={p[1]} r={12} fill="#1f3b6e" stroke="#fff" strokeWidth={1.5} />
          <text x={p[0]} y={p[1] + 26} fill="#93a0bd" fontSize={10} textAnchor="middle">#{a}</text></g>); })}
      </svg>
      <p className="small muted">🔴 arrows = real money movement · labels = ₹ per transfer. <b>Real transaction graph (AMLSim)</b> — what SENTINEL does given bank link data (Phase-2); recovers planted rings at 100% purity.</p>
    </>
  );
}

export default function Network() {
  const [rings, setRings] = useState<RingsResp | null>(null);
  const [flow, setFlow] = useState<FlowResp | null>(null);
  const [rows, setRows] = useState<AccountRow[] | null>(null);
  useEffect(() => { api.rings().then(setRings); api.flow().then(setFlow); api.accounts().then((d) => setRows(d.accounts)); }, []);
  if (!rings || !flow || !rows) return <Loading what="network maps" />;
  const rv = rings.rings;
  return (
    <>
      <h1 className="h1">Network & Money-Flow</h1>
      <p className="sub">Three lenses on the network behind an account: its live look-alike map (any account), validated candidate rings, and real money-flow on transaction data.</p>
      {rv && (
        <div className="grid g4">
          <Metric k="Candidate rings" v={rv.n_candidate_rings} />
          <Metric k="Mules grouped" v={`${rv.mules_in_rings}/${rv.n_mules}`} />
          <Metric k="Ring #1 stability" v={(rv.validation?.ring1_subsample_stability_jaccard ?? 0).toFixed(2)} d="Jaccard under subsampling" />
          <Metric k="Tighter than chance" v={`~${Math.round((rv.validation?.ring1_intra_sim ?? 0) / Math.max(rv.validation?.legit_subset_sim_mean ?? 1e-9, 1e-9))}×`} d="vs random legit" />
        </div>
      )}
      <div className="card" style={{ marginTop: 16 }}>
        <h3>🗺️ Account network map — live, any account (incl. uploaded)</h3>
        <AccountMap rows={rows} />
      </div>
      <div className="grid g2" style={{ marginTop: 16 }}>
        <Card title="🕸️ Validated candidate rings (known mules)"><RingGraph data={rings} /></Card>
        <Card title="💸 Real money-flow — transaction graph (AMLSim)"><FlowGraph data={flow} /></Card>
      </div>
    </>
  );
}
