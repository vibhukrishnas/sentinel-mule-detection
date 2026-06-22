import { useEffect, useMemo, useState } from "react";
import { api, AccountRow, Summary } from "../api";
import { Metric, Card, Loading, fmtInr } from "../ui";
import { BarChart, Bar as RBar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function Analytics({ s, threshold, setThreshold }: { s: Summary; threshold: number; setThreshold: (n: number) => void }) {
  const [rows, setRows] = useState<AccountRow[]>([]);
  const thr = threshold;
  const setThr = setThreshold;
  const [muleLoss, setMuleLoss] = useState(250000);
  const [reviewCost, setReviewCost] = useState(400);
  const [fpHarm, setFpHarm] = useState(25000);
  const [analysts, setAnalysts] = useState(20);
  const [mins, setMins] = useState(15);
  useEffect(() => { api.accounts().then((d) => setRows(d.accounts)); }, []);

  const labeled = rows.filter((r) => r.ground_truth != null);
  const impact = useMemo(() => {
    const flagged = labeled.filter((r) => r.probability >= thr);
    const tp = flagged.filter((r) => r.ground_truth === "MULE").length;
    const fp = flagged.length - tp;
    const fn = labeled.filter((r) => r.ground_truth === "MULE" && r.probability < thr).length;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const prec = flagged.length ? tp / flagged.length : 0;
    const net = tp * muleLoss - flagged.length * reviewCost - fp * fpHarm;
    return { alerts: flagged.length, tp, fp, fn, recall, prec, net };
  }, [labeled, thr, muleLoss, reviewCost, fpHarm]);

  const dist = useMemo(() => {
    const b: Record<number, number> = {};
    rows.forEach((r) => { const k = Math.floor(r.risk_score / 10) * 10; b[k] = (b[k] ?? 0) + 1; });
    return Array.from({ length: 11 }, (_, i) => ({ bucket: i * 10, n: b[i * 10] ?? 0 }));
  }, [rows]);

  const tLo = s.abstention?.t_lo ?? 0.1;
  const priority = rows.filter((r) => r.probability > tLo).length;
  const autoClear = rows.length - priority;
  const dailyCap = Math.round((analysts * 8 * 60) / mins);

  if (!rows.length) return <Loading what="analytics" />;
  return (
    <>
      <h1 className="h1">Analytics</h1>
      <p className="sub">Live threshold economics + alert-fatigue reduction. Plug in your bank's real numbers.</p>

      <Card title={`🎚️ Operating threshold = ${thr.toFixed(2)}`}>
        <input type="range" min={0.01} max={0.99} step={0.01} value={thr}
          onChange={(e) => setThr(Number(e.target.value))} style={{ width: "100%" }} />
        {labeled.length ? (
          <div className="grid g4" style={{ marginTop: 14 }}>
            <Metric k="Alerts raised" v={impact.alerts} />
            <Metric k="Mules caught" v={`${impact.tp}/${impact.tp + impact.fn}`} d={`recall ${(impact.recall * 100).toFixed(0)}%`} cls="good" />
            <Metric k="False alarms" v={impact.fp} d={`precision ${(impact.prec * 100).toFixed(0)}%`} />
            <Metric k="Net ₹ impact" v={fmtInr(impact.net)} cls={impact.net >= 0 ? "good" : "bad"} />
          </div>
        ) : <p className="muted">no labels in sample</p>}
        <div className="row" style={{ marginTop: 14 }}>
          <label className="small muted">Mule loss ₹<br /><input type="number" value={muleLoss} onChange={(e) => setMuleLoss(+e.target.value)} /></label>
          <label className="small muted">Review ₹<br /><input type="number" value={reviewCost} onChange={(e) => setReviewCost(+e.target.value)} /></label>
          <label className="small muted">False-freeze harm ₹<br /><input type="number" value={fpHarm} onChange={(e) => setFpHarm(+e.target.value)} /></label>
        </div>
      </Card>

      <div className="grid g2" style={{ marginTop: 16 }}>
        <Card title="Risk-score distribution">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={dist} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
              <XAxis dataKey="bucket" stroke="#93a0bd" fontSize={12} />
              <YAxis stroke="#93a0bd" fontSize={12} />
              <Tooltip contentStyle={{ background: "#141b2e", border: "1px solid #263150", borderRadius: 10, color: "#e8edf7" }} />
              <RBar dataKey="n" fill="#4f7cff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="📉 Alert-fatigue & analyst-capacity optimizer">
          <div className="grid g2">
            <Metric k="Without prioritization" v={rows.length} d="every account" />
            <Metric k="With SENTINEL" v={priority} d={`-${Math.round((1 - priority / rows.length) * 100)}% workload`} cls="good" />
          </div>
          <p className="small muted" style={{ margin: "8px 0" }}>{autoClear} confidently-legit accounts auto-cleared (p ≤ {tLo.toFixed(2)}); only the rest reach a human.</p>
          <div className="row">
            <label className="small muted">Analysts<br /><input type="number" value={analysts} onChange={(e) => setAnalysts(+e.target.value)} /></label>
            <label className="small muted">Mins/review<br /><input type="number" value={mins} onChange={(e) => setMins(+e.target.value)} /></label>
            <Metric k="Daily capacity" v={`${dailyCap.toLocaleString()} cases`} d={`${analysts} × 8h ÷ ${mins}m`} />
          </div>
        </Card>
      </div>
    </>
  );
}
