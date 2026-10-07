
import { useEffect, useState } from "react";
import { api } from "./api";
import Home from "./components/Home";
import StepNav from "./components/StepNav";
import PaymentSimulator from "./components/PaymentSimulator";
import BatchUpload from "./components/BatchUpload";
import Report from "./components/Report";
import Performance from "./components/Performance";
 
// Pages in order: Home -> Make Payment -> Batch Upload -> Investigation Report -> Model Performance
const PAGES = [
  ["home", "Home"],
  ["pay", "Make Payment"],
  ["batch", "Batch Upload"],
  ["report", "Investigation Report"],
  ["performance", "Model Performance"],
];
const keys = PAGES.map(([k]) => k);
const label = (k) => PAGES.find(([key]) => key === k)[1];
const fromHash = () => {
  const k = window.location.hash.replace("#/", "");
  return keys.includes(k) ? k : "home";
};
 
export default function App() {
  const [page, setPage] = useState(fromHash());
  const [health, setHealth] = useState(null); // null = loading, {error} = backend down
 
  // Keep the URL hash and page in sync so the browser Back button works
  useEffect(() => {
    const onHash = () => setPage(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
 
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ error: true }));
  }, []);
 
  const go = (k) => {
    window.location.hash = `/${k}`;
    window.scrollTo(0, 0);
  };
  const i = keys.indexOf(page);
  const prev = keys[i - 1];
  const next = keys[i + 1];
 
  return (
    <div className="app">
      {page !== "home" && (
        <header>
          <h1 className="brand" onClick={() => go("home")}>Smart Fraud Detection System</h1>
          <nav>
            {PAGES.slice(1).map(([key, name], n) => (
              <button key={key} className={page === key ? "active" : ""} onClick={() => go(key)}>
                {n + 1}. {name}
              </button>
            ))}
          </nav>
        </header>
      )}
 
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
 
      <main>
        {page === "home" && <Home onStart={() => go("pay")} />}
        {page === "pay" && <PaymentSimulator />}
        {page === "batch" && <BatchUpload />}
        {page === "report" && <Report />}
        {page === "performance" && <Performance />}
      </main>
 
      {page !== "home" && (
        <StepNav
          onBack={() => go(prev)}
          backLabel={label(prev)}
          onNext={next ? () => go(next) : null}
          nextLabel={next ? label(next) : ""}
        />
      )}
    </div>
  );
}
 
