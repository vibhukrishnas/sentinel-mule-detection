import React from "react";

export const Metric = ({ k, v, d, cls }: { k: string; v: React.ReactNode; d?: React.ReactNode; cls?: string }) => (
  <div className="metric">
    <div className="k">{k}</div>
    <div className="v">{v}</div>
    {d && <div className={`d ${cls ?? "muted"}`}>{d}</div>}
  </div>
);

export const Card = ({ title, children }: { title?: string; children: React.ReactNode }) => (
  <div className="card">{title && <h3>{title}</h3>}{children}</div>
);

export const Badge = ({ band }: { band: string }) => <span className={`badge b-${band}`}>{band}</span>;

export const Bar = ({ pct }: { pct: number }) => (
  <div className="bar"><i style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} /></div>
);

export const Loading = ({ what }: { what?: string }) => <div className="loading">Loading {what ?? "…"}</div>;

export const fmtInr = (n: number) =>
  n >= 1e7 ? `₹${(n / 1e7).toFixed(2)} Cr` : n >= 1e5 ? `₹${(n / 1e5).toFixed(2)} L` : `₹${n.toLocaleString("en-IN")}`;
