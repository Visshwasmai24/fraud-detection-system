import { useEffect, useState } from "react";
import { api, inr, pct } from "../api";
import RiskBadge from "./RiskBadge";

export default function Report() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");

  const load = () => api.report().then(setRows).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  const decide = async (id, decision) => {
    try {
      const comment = decision === "pending" ? "" : (window.prompt("Optional analyst comment:", "") ?? "");
      await api.decide(id, decision, comment); load();
    } catch (e) { setError(e.message); }
  };
  const clear = async () => {
    if (window.confirm("Remove all cases from the report?")) { await api.clearReport(); load(); }
  };

  return (
    <section className="card">
      <div className="row between">
        <div>
          <h2>Investigation report</h2>
          <p className="muted">Medium and High risk transactions, ranked by fraud risk score. Record your decision for each
            case — confirmed outcomes are stored as labelled feedback for controlled future retraining; the current model is never replaced automatically.</p>
        </div>
        <div className="row">
          <a className="btn" href="/api/report/download">Download CSV</a>
          <button onClick={load}>Refresh</button>
          <button className="danger" onClick={clear}>Clear</button>
        </div>
      </div>
      {error && <div className="banner bad">{error}</div>}
      {rows && rows.length === 0 && <p className="muted">No flagged transactions yet. Make a payment or upload a file first.</p>}
      {rows && rows.length > 0 && (
        <div className="scroll">
          <table>
            <thead><tr><th>#</th><th>Transaction</th><th>Risk</th><th>Risk score</th><th>Amount</th><th>Type</th>
              <th>Why flagged</th><th>Suggested action</th><th>Analyst decision</th><th>Comment</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.transaction_id}>
                  <td>{r.rank}</td><td>{r.transaction_id}</td><td><RiskBadge level={r.risk_level} /></td>
                  <td>{pct(r.risk_score)}</td><td>{inr(r.amount)}</td><td>{r.txn_type}</td>
                  <td className="small">{r.reasons.join(" · ")}</td><td className="small">{r.suggested_action}</td>
                  <td>
                    <select value={r.analyst_decision} onChange={(e) => decide(r.transaction_id, e.target.value)}>
                      <option value="pending">Pending</option>
                      <option value="confirmed_fraud">Confirmed fraud</option>
                      <option value="genuine">Genuine</option>
                    </select>
                    </td>
                    <td className="small muted">{r.analyst_comment || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
