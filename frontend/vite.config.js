import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the React app (port 5173) forwards /api calls to Flask (port 5000),
// so the browser never hits a cross-origin problem.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:5000" } },
});
