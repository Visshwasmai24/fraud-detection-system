// Thin wrapper around the Flask API.
//
// Local development:
// Vite forwards /api/... requests to http://localhost:5000.
//
// Deployment:
// VITE_API_URL points to the deployed Flask backend on Render.

const API_URL = import.meta.env.VITE_API_URL || "";

async function handle(res) {
  let data = null;

  try {
    data = await res.json();
  } catch {
    // Response was not JSON
  }

  if (!res.ok) {
    throw new Error(
      data?.error ||
      data?.hint ||
      `Request failed (${res.status})`
    );
  }

  return data;
}

export const api = {
  // Health check
  health: () =>
    fetch(`${API_URL}/api/health`).then(handle),

  // Single transaction prediction
  predict: (txn) =>
    fetch(`${API_URL}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(txn),
    }).then(handle),

  // Batch CSV upload
  batch: (file) => {
    const form = new FormData();
    form.append("file", file);

    return fetch(`${API_URL}/api/batch`, {
      method: "POST",
      body: form,
    }).then(handle);
  },

  // Fraud/risk report
  report: () =>
    fetch(`${API_URL}/api/report`).then(handle),

  // Approve/reject a report case
  decide: (id, decision, comment = "") =>
    fetch(
      `${API_URL}/api/report/${encodeURIComponent(id)}/decision`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          decision,
          comment,
        }),
      }
    ).then(handle),

  // Clear report
  clearReport: () =>
    fetch(`${API_URL}/api/report`, {
      method: "DELETE",
    }).then(handle),

  // Model performance metrics
  metrics: () =>
    fetch(`${API_URL}/api/metrics`).then(handle),
};

// Format amount as Indian Rupees
export const inr = (n) =>
  "₹" +
  Number(n).toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  });

// Format decimal as percentage
export const pct = (x) =>
  (x * 100).toFixed(1) + "%";
