import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backend = "http://localhost:8020";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/search": backend,
      "/images": backend,
      "/health": backend,
    },
  },
});
