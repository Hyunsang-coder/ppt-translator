import type { NextConfig } from "next";

// When building for the Tauri desktop app we need a fully static export
// (no Node server). The web/EC2 build keeps its default behaviour.
const isTauri = process.env.TAURI_BUILD === "1";

const nextConfig: NextConfig = {
  ...(isTauri
    ? {
        output: "export",
        // Tauri serves files from disk; unoptimized images avoid the Next.js
        // image optimizer which requires a server.
        images: { unoptimized: true },
      }
    : {}),
};

export default nextConfig;
