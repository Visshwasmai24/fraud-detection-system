import { useState } from "react";
import { api, inr, pct } from "../api";
import RiskBadge from "./RiskBadge";

export default function BatchUpload() {
  const [file, setFile] = useState(null);
  const [res, setRes] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (!file) return;
    setBusy(true); setError(""); setRes(null);
    try { setRes(await api.batch(file)); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  const s = res?.summary;
  return (
    <section className="card">
      <h2>Batch upload</h2>
      <p className="muted">
        Upload transactions as <b>CSV</b>, a <b>JSON / JSON-Lines</b> log (nested fields), or a CSV that uses the
        common column names. The Data Adapter converts it to one common format before scoring.
      </p>
      <div className="row">
        <input type="file" accept=".csv,.json,.jsonl,.ndjson" onChange={(e) => setFile(e.target.files[0])} />
        <button className="primary" disabled={!file || busy} onClick={run}>{busy ? "Scoring…" : "Score file"}</button>
      </div>
      {error && <div className="banner bad">{error}</div>}

      {s && (
        <>
          <div className="stats">
            <Stat label="Rows scored" value={s.rows.toLocaleString()} />
            <Stat label="Low" value={s.risk_counts.Low.toLocaleString()} cls="low" />
            <Stat label="Medium" value={s.risk_counts.Medium.toLocaleString()} cls="medium" />
            <Stat label="High" value={s.risk_counts.High.toLocaleString()} cls="high" />
            <Stat label="Source detected" value={s.data_source} />
          </div>
          {s.notes.map((n, i) => <div key={i} className="banner warn small">{n}</div>)}
          {s.quick_metrics && (
            <p>The file contained fraud labels. Against them: precision <b>{pct(s.quick_metrics.precision)}</b>, recall{" "}
              <b>{pct(s.quick_metrics.recall)}</b> (TP {s.quick_metrics.tp}, FP {s.quick_metrics.fp}, FN {s.quick_metrics.fn}).
              <span className="muted"> If this file was part of the training data, these numbers are optimistic.</span></p>
          )}
          <p className="muted">{s.added_to_report} flagged transactions were added to the Investigation Report. Top 50 shown:</p>
          <div className="scroll">
            <table>
              <thead><tr><th>Transaction</th><th>Risk</th><th>Risk score</th><th>Amount</th><th>Type</th><th>Why flagged</th></tr></thead>
              <tbody>
                {res.top_flagged.map((r) => (
                  <tr key={r.transaction_id}>
                    <td>{r.transaction_id}</td><td><RiskBadge level={r.risk_level} /></td>
                    <td>{pct(r.risk_score)}</td><td>{inr(r.amount)}</td><td>{r.txn_type}</td>
                    <td className="small">{r.reasons.map((x) => x.text).join(" · ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

function Stat({ label, value, cls = "" }) {
  return <div className={`stat ${cls}`}><div className="v">{value}</div><div className="l">{label}</div></div>;
}
