// Thin wrapper around the Flask API. All paths are relative (/api/...),
// Vite forwards them to http://localhost:5000 while developing.

async function handle(res) {
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(data?.error || data?.hint || `Request failed (${res.status})`);
  return data;
}

export const api = {
  health: () => fetch("/api/health").then(handle),
  predict: (txn) =>
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(txn),
    }).then(handle),
  batch: (file) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/batch", { method: "POST", body: form }).then(handle);
  },
  report: () => fetch("/api/report").then(handle),
  decide: (id, decision, comment = "") =>
    fetch(`/api/report/${encodeURIComponent(id)}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, comment }),
    }).then(handle),
  clearReport: () => fetch("/api/report", { method: "DELETE" }).then(handle),
  metrics: () => fetch("/api/metrics").then(handle),
};

export const inr = (n) => "₹" + Number(n).toLocaleString("en-IN", { maximumFractionDigits: 2 });
export const pct = (x) => (x * 100).toFixed(1) + "%";
