import { useEffect, useState } from "react";
import { api } from "./api";
import PaymentSimulator from "./components/PaymentSimulator";
import BatchUpload from "./components/BatchUpload";
import Report from "./components/Report";
import Performance from "./components/Performance";

const TABS = [
  ["pay", "Make Payment"],
  ["batch", "Batch Upload"],
  ["report", "Investigation Report"],
  ["perf", "Model Performance"],
];

export default function App() {
  const [tab, setTab] = useState("pay");
  const [health, setHealth] = useState(null); // null = loading, {error} = backend down

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ error: true }));
  }, []);

  return (
    <div className="app">
      <header>
        <div>
          <h1>Smart Fraud Detection System</h1>
          <p>Online payment fraud detection with explainable risk scoring</p>
        </div>
        <nav>
          {TABS.map(([key, label]) => (
            <button key={key} className={tab === key ? "active" : ""} onClick={() => setTab(key)}>
              {label}
            </button>
          ))}
        </nav>
      </header>

      {health?.error && (
        <div className="banner bad">
          Cannot reach the backend. Start it with <code>python -m backend.app</code> (port 5000).
        </div>
      )}
      {health && !health.error && !health.model_loaded && (
        <div className="banner warn">
          No trained model yet. Run <code>python -m backend.generate_data</code> and then{" "}
          <code>python -m backend.train</code>, then refresh this page.
        </div>
      )}
      {health?.model_loaded && (
        <div className="banner ok">
          Model ready · <b>{health.model_version || "XGBoost"}</b> · calibrated risk scoring · trained on <b>{health.trained_on}</b> data · {health.trained_at}
        </div>
      )}

      <main>
        {tab === "pay" && <PaymentSimulator />}
        {tab === "batch" && <BatchUpload />}
        {tab === "report" && <Report />}
        {tab === "perf" && <Performance />}
      </main>
    </div>
  );
}
