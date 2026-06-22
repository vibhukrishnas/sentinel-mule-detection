import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxies the API paths to the FastAPI backend on :8000 (same-origin in prod, where
// FastAPI serves the built UI itself). Keep this list in sync with src/api.ts endpoints.
const API = ["/summary", "/accounts", "/account", "/rings", "/flow", "/network", "/upload", "/reset", "/health", "/score", "/report"];
const proxy = Object.fromEntries(API.map((p) => [p, { target: "http://localhost:8000", changeOrigin: true }]));

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
});
