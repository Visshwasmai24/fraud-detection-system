import { useEffect, useState } from "react";
import { api, pct } from "../api";

export default function Performance() {
  const [m, setM] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { api.metrics().then(setM).catch((e) => setError(e.message)); }, []);

  if (error) return <div className="banner warn">{error}</div>;
  if (!m) return <p className="muted">Loading…</p>;

  const d = m.at_decision_threshold;
  const cm = d.confusion_matrix;
  const maxShap = m.shap_importance[0].mean_abs_shap;

  return (
    <div className="stack">
      <section className="card">
        <h2>Test-set results</h2>
        <p className="muted">
          Data: <b>{m.data_source}</b> · {m.rows.toLocaleString()} transactions ({m.fraud_rows.toLocaleString()} fraud) ·
          evaluated on {m.test_rows.toLocaleString()} later test transactions ({m.test_fraud} fraud). SMOTE is applied inside training folds only.
          The primary split is chronological so the model learns from the past and is tested on later transactions.
        </p>
        <div className="stats">
          <Stat label="Precision" value={pct(d.precision)} />
          <Stat label="Recall" value={pct(d.recall)} />
          <Stat label="F1-score" value={pct(d.f1)} />
          <Stat label="PR-AUC" value={m.pr_auc.toFixed(3)} />
          <Stat label="ROC-AUC" value={m.roc_auc.toFixed(3)} />
          <Stat label="Accuracy" value={pct(d.accuracy)} />
        </div>
        {m.notes.map((n, i) => <div key={i} className="banner warn small">{n}</div>)}
        <p className="muted small">Decision threshold {d.threshold} selected with a cost-sensitive policy (FN cost {m.threshold_policy.false_negative_cost}× FP cost {m.threshold_policy.false_positive_cost}) using training-period out-of-fold scores. ·
          Tuned with {m.cv.n_iter} random configs × {m.cv.folds}-fold CV, scoring = {m.cv.scoring}. ·
          Risk score is calibrated; it is an estimated risk probability, not a guarantee of fraud.</p>
      </section>

      <div className="grid two">
        <section className="card">
          <h2>Threshold analysis</h2>
          <p className="muted small">The operating threshold is selected on training-period out-of-fold predictions using a higher cost for missed fraud. The final test set remains untouched.</p>
          <table>
            <thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Flagged</th></tr></thead>
            <tbody>
              {m.threshold_analysis.map((r) => (
                <tr key={r.threshold}><td>{r.threshold}</td><td>{pct(r.precision)}</td><td>{pct(r.recall)}</td><td>{pct(r.f1)}</td><td>{r.flagged}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
        <section className="card">
          <h2>Confusion matrix</h2>
          <table className="cm">
            <thead><tr><th></th><th>Predicted genuine</th><th>Predicted fraud</th></tr></thead>
            <tbody>
              <tr><th>Actual genuine</th><td>{cm.tn.toLocaleString()}</td><td className="bad">{cm.fp.toLocaleString()} <small>false alarms</small></td></tr>
              <tr><th>Actual fraud</th><td className="bad">{cm.fn.toLocaleString()} <small>missed</small></td><td>{cm.tp.toLocaleString()}</td></tr>
            </tbody>
          </table>
          <h3>Risk bands vs. reality</h3>
          <table>
            <thead><tr><th>Band</th><th>Genuine</th><th>Fraud</th></tr></thead>
            <tbody>
              {["Low", "Medium", "High"].map((k) => [k, m.risk_band_distribution[k]]).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{v.genuine.toLocaleString()}</td><td>{v.fraud.toLocaleString()}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>

      <section className="card">
        <h2>What drives the model (SHAP)</h2>
        {m.shap_importance.slice(0, 10).map((s) => (
          <div className="shaprow" key={s.feature}>
            <span>{s.feature}</span>
            <div className="bar"><div style={{ width: `${(s.mean_abs_shap / maxShap) * 100}%` }} /></div>
            <span className="small">{s.mean_abs_shap.toFixed(2)}</span>
          </div>
        ))}
      </section>

      <div className="grid two">
        <section className="card"><h2>Precision–recall curve</h2><img src="/api/plots/pr_curve.png" alt="PR curve" /></section>
        <section className="card"><h2>Confusion matrix</h2><img src="/api/plots/confusion_matrix.png" alt="Confusion matrix" /></section>
      </div>


      {m.temporal_evaluation && (
        <section className="card">
          <h2>Temporal vs random diagnostic</h2>
          <p className="muted small">The chronological evaluation above is the primary result. This secondary random holdout is shown only to demonstrate how a random split can give a different estimate.</p>
          <div className="stats">
            <Stat label="Random PR-AUC" value={m.temporal_evaluation.pr_auc.toFixed(3)} />
            <Stat label="Random ROC-AUC" value={m.temporal_evaluation.roc_auc.toFixed(3)} />
            <Stat label="Random F1" value={pct(m.temporal_evaluation.metrics.f1)} />
            <Stat label="Random recall" value={pct(m.temporal_evaluation.metrics.recall)} />
          </div>
        </section>
      )}
      {m.drift && (
        <section className="card">
          <h2>Drift monitoring</h2>
          <p className="muted">Current status: <b>{m.drift.status}</b> · high-drift features: {m.drift.high_drift_features} · moderate: {m.drift.moderate_drift_features}</p>
        </section>
      )}

      {m.generalization ? (
        <section className="card">
          <h2>Unseen fraud-pattern test</h2>
          <p className="muted">
            Each row removes one fraud pattern from training, then checks how much of it the model still catches (threshold{" "}
            {m.generalization.threshold}). <b>This is the honest limit of a supervised model:</b> fraud that looks nothing like
            the training examples is largely missed. That is why the design includes monitoring and periodic retraining on
            analyst-confirmed cases.
          </p>
          <table>
            <thead><tr><th>Held-out pattern</th><th>Cases</th><th>Recall (seen in training)</th><th>Recall (never seen)</th></tr></thead>
            <tbody>
              {m.generalization.results.map((r) => (
                <tr key={r.pattern}><td>{r.pattern}</td><td>{r.test_fraud_cases}</td><td>{pct(r.recall_seen)}</td><td><b>{pct(r.recall_unseen)}</b></td></tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : (
        <section className="card"><h2>Unseen fraud-pattern test</h2>
          <p className="muted">Not run yet. Run <code>python -m backend.generalization_test</code> and refresh.</p></section>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return <div className="stat"><div className="v">{value}</div><div className="l">{label}</div></div>;
}
