import type { NextConfig } from "next";

// When building for the Tauri desktop app we need a fully static export
// (no Node server). The public web build keeps the default Next.js output and
// only serves the desktop download 안내 page.
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
