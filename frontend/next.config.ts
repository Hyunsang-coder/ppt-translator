import type { NextConfig } from "next";

// When building for the Tauri desktop app we need a fully static export
// (no Node server). The public web build keeps the default Next.js output and
// only serves the desktop download 안내 page.
const isTauri = process.env.TAURI_BUILD === "1";
const isPublicWebBuild = !isTauri && process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  ...(isTauri
    ? {
        output: "export",
        // Tauri serves files from disk; unoptimized images avoid the Next.js
        // image optimizer which requires a server.
        images: { unoptimized: true },
      }
    : isPublicWebBuild
      ? {
          async redirects() {
            return [
              { source: "/translate", destination: "/", permanent: false },
              { source: "/extract", destination: "/", permanent: false },
              { source: "/settings", destination: "/", permanent: false },
            ];
          },
        }
      : {}),
};

export default nextConfig;
