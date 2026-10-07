
export default function Home({ onStart }) {
  return (
    <section className="hero">
      <h1>Smart Fraud Detection System</h1>
      <p className="hero-desc">
        Detects fraud in online payments in real time. Every transaction is scored by a tuned
        XGBoost model, rated Low, Medium or High risk, and explained in plain English so you
        know exactly why it was flagged.
      </p>
      <p className="hero-flow">Allows you to Make Payment &nbsp;→&nbsp; Batch Upload &nbsp;→&nbsp; Investigation Report</p>
      <button className="primary hero-btn" onClick={onStart}>Get Started</button>
    </section>
  );
}
 
