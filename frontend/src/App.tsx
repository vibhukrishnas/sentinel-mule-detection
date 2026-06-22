import { useEffect, useRef, useState } from "react";
import { api, Summary } from "./api";
import { Loading } from "./ui";
import Overview from "./views/Overview";
import Investigate from "./views/Investigate";
import Network from "./views/Network";
import Triage from "./views/Triage";
import Analytics from "./views/Analytics";

const TABS = [
  { id: "overview", label: "📊 Overview" },
  { id: "investigate", label: "🔍 Investigate" },
  { id: "network", label: "🕸️ Network & Money-Flow" },
  { id: "triage", label: "🚦 Triage Queue" },
  { id: "analytics", label: "📈 Analytics" },
];

export default function App() {
  const [tab, setTab] = useState("overview");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [version, setVersion] = useState(0);          // bump to refetch population
  const [source, setSource] = useState("connecting…");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadSummary = () => api.summary().then((s) => { setSummary(s); setSource(s.source); })
    .catch((e) => setErr("Cannot reach the SENTINEL API. Run:  uvicorn src.api:app --port 8000  (" + e.message + ")"));
  useEffect(() => { loadSummary(); }, []);

  const onUpload = async (f: File) => {
    setBusy(true); setErr(null);
    try {
      const r = await api.upload(f);
      setSource(r.source); setVersion((v) => v + 1); await loadSummary();
    } catch (e: any) { setErr("Upload failed — " + e.message); }
    finally { setBusy(false); }
  };
  const onReset = async () => { setBusy(true); await api.reset(); setVersion((v) => v + 1); await loadSummary(); setBusy(false); };

  return (
    <div className="app">
      <aside className="sidebar" style={{ display: "flex", flexDirection: "column" }}>
        <div className="brand">🛡️ SENTINEL<small>Mule-Account Risk · Command Center</small></div>
        <nav className="nav">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>{t.label}</button>
          ))}
        </nav>
        <div className="src">
          <span className="tag">{source}</span>
          <p className="small muted" style={{ marginTop: 12 }}>
            Live LightGBM scoring via FastAPI. F3912 leakage excluded — honest numbers.
          </p>
        </div>
      </aside>

      <main className="main">
        {/* command bar: data source + upload + global threshold */}
        <div className="cmdbar">
          <div className="row" style={{ alignItems: "center", gap: 10 }}>
            <input ref={fileRef} type="file" accept=".csv" style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
            <button className="primary" onClick={() => fileRef.current?.click()}>⬆ Upload CSV</button>
            <button className="ghost" onClick={onReset}>↺ Reset</button>
            {busy && <span className="muted small">scoring…</span>}
          </div>
          <div className="thr">
            <span className="small muted">Alert threshold</span>
            <input type="range" min={0.01} max={0.99} step={0.01} value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))} />
            <b>{threshold.toFixed(2)}</b>
          </div>
        </div>

        {err && <div className="banner err">{err}</div>}
        {!summary && !err && <Loading what="command center" />}
        {summary && (
          <div key={version}>
            {tab === "overview" && <Overview s={summary} />}
            {tab === "investigate" && <Investigate threshold={threshold} />}
            {tab === "network" && <Network />}
            {tab === "triage" && <Triage s={summary} threshold={threshold} />}
            {tab === "analytics" && <Analytics s={summary} threshold={threshold} setThreshold={setThreshold} />}
          </div>
        )}
      </main>
    </div>
  );
}
