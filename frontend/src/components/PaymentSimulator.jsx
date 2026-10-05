import { useState } from "react";
import { api, inr, pct } from "../api";
import RiskBadge from "./RiskBadge";

// Demo scenarios (the college-fee ones show the "online fee payment" use case)
const PRESETS = {
  "Normal payment": {
    amount: 650, type: "PAYMENT", time: "14:10", balance: 18000, age: 900, txns: 2, failed: 0,
    newReceiver: false, receiverBalance: 50000, newDevice: false, locationDiff: false,
  },
  "College fee (genuine)": {
    amount: 85000, type: "PAYMENT", time: "11:00", balance: 120000, age: 1200, txns: 1, failed: 0,
    newReceiver: false, receiverBalance: 900000, newDevice: false, locationDiff: false,
  },
  "College fee (fake portal)": {
    amount: 85000, type: "TRANSFER", time: "02:30", balance: 120000, age: 1200, txns: 4, failed: 2,
    newReceiver: true, receiverBalance: 400, newDevice: true, locationDiff: true,
  },
  "Large transfer to new person": {
    amount: 39000, type: "TRANSFER", time: "15:00", balance: 60000, age: 800, txns: 2, failed: 0,
    newReceiver: true, receiverBalance: 30000, newDevice: false, locationDiff: false,
  },
  "Account drain": {
    amount: 46000, type: "CASH_OUT", time: "03:20", balance: 48000, age: 400, txns: 3, failed: 1,
    newReceiver: true, receiverBalance: 250, newDevice: true, locationDiff: false,
  },
  "Rapid small payments": {
    amount: 900, type: "TRANSFER", time: "19:45", balance: 15000, age: 500, txns: 14, failed: 3,
    newReceiver: true, receiverBalance: 20000, newDevice: true, locationDiff: false,
  },
};

// Build the nested JSON transaction the backend's Data Adapter expects
function toPayload(f) {
  const d = new Date();
  const [h, m] = f.time.split(":");
  d.setHours(+h, +m, 0, 0);
  const pad = (n) => String(n).padStart(2, "0");
  const ts = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${f.time}:00`;
  return {
    timestamp: ts,
    amount: Number(f.amount),
    type: f.type,
    sender: {
      balance_before: Number(f.balance),
      account_age_days: Number(f.age),
      txn_count_24h: Number(f.txns),
      failed_attempts: Number(f.failed),
    },
    receiver: { balance_before: Number(f.receiverBalance), is_new_receiver: f.newReceiver },
    device_info: { is_new_device: f.newDevice },
    location: { geo_mismatch: f.locationDiff },
  };
}

export default function PaymentSimulator() {
  const [form, setForm] = useState(PRESETS["Normal payment"]);
  const [preset, setPreset] = useState("Normal payment");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const set = (k) => (e) => {
    const v = e.target.type === "checkbox" ? e.target.checked : e.target.value;
    setForm({ ...form, [k]: v });
    setPreset("");
  };

  const pick = (name) => { setForm(PRESETS[name]); setPreset(name); setResult(null); setError(""); };

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setError(""); setResult(null);
    try { setResult(await api.predict(toPayload(form))); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };

  return (
    <div className="grid two">
      <section className="card">
        <h2>UPI-style payment simulator</h2>
        <p className="muted">Pick a demo scenario or edit the fields, then press Pay. The transaction goes through
          the Data Adapter, feature engineering and the tuned XGBoost model.</p>

        <div className="chips">
          {Object.keys(PRESETS).map((n) => (
            <button key={n} type="button" className={`chip ${preset === n ? "on" : ""}`} onClick={() => pick(n)}>{n}</button>
          ))}
        </div>

        <p className="muted small">Demo scenarios: exact scores depend on the model you trained, so a result near a
          risk-band edge may differ slightly.</p>

        <form onSubmit={submit} className="form">
          <label>Amount (₹)<input type="number" min="1" step="any" required value={form.amount} onChange={set("amount")} /></label>
          <label>Payment type
            <select value={form.type} onChange={set("type")}>
              <option>PAYMENT</option><option>TRANSFER</option><option>CASH_OUT</option>
            </select>
          </label>
          <label>Time of payment<input type="time" required value={form.time} onChange={set("time")} /></label>
          <label>Sender balance (₹)<input type="number" min="1" step="any" required value={form.balance} onChange={set("balance")} /></label>
          <label>Account age (days)<input type="number" min="0" required value={form.age} onChange={set("age")} /></label>
          <label>Transactions in last 24h<input type="number" min="0" required value={form.txns} onChange={set("txns")} /></label>
          <label>Failed attempts before this<input type="number" min="0" required value={form.failed} onChange={set("failed")} /></label>
          <label>Receiver balance (₹)<input type="number" min="0" step="any" required value={form.receiverBalance} onChange={set("receiverBalance")} /></label>
          <label className="check"><input type="checkbox" checked={form.newReceiver} onChange={set("newReceiver")} /> First time paying this receiver</label>
          <label className="check"><input type="checkbox" checked={form.newDevice} onChange={set("newDevice")} /> Payment from a new device</label>
          <label className="check"><input type="checkbox" checked={form.locationDiff} onChange={set("locationDiff")} /> Location differs from usual</label>
          <button className="primary" disabled={busy}>{busy ? "Checking…" : `Pay ${inr(form.amount || 0)}`}</button>
        </form>
        {error && <div className="banner bad">{error}</div>}
      </section>

      <section className="card">
        <h2>Fraud check result</h2>
        {!result && <p className="muted">Result appears here after you press Pay.</p>}
        {result && (
          <div className={`result ${result.risk_level.toLowerCase()}`}>
            <div className="result-top">
              <RiskBadge level={result.risk_level} />
              <span className="prob">{pct(result.risk_score)} fraud risk score</span>
            </div>
            <div className="bar"><div style={{ width: pct(result.risk_score) }} /></div>
            <p><b>Suggested action:</b> {result.suggested_action}</p>
            <h3>{result.reasons_label}</h3>
            {result.reasons.length === 0 ? <p className="muted">None</p> : (
              <ul>{result.reasons.map((r, i) => <li key={i}>{r.text}</li>)}</ul>
            )}
            <p className="muted small">Transaction {result.transaction_id}
              {result.risk_level !== "Low" && " · added to the Investigation Report"}</p>
          </div>
        )}
      </section>
    </div>
  );
}
