"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { isTauri } from "@/lib/api-base";
import { SidecarProvider } from "@/components/sidecar-provider";

const LOCAL_WEB_HOSTS = new Set(["localhost", "127.0.0.1", "0.0.0.0", "::1"]);

function isHostedWebBuild(): boolean {
  if (typeof window === "undefined" || isTauri()) return false;
  return !LOCAL_WEB_HOSTS.has(window.location.hostname);
}

export function DesktopShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [redirecting, setRedirecting] = useState(false);

  useEffect(() => {
    const hostedWebBuild = isHostedWebBuild();
    setRedirecting(hostedWebBuild);
    setReady(true);

    if (hostedWebBuild) {
      router.replace("/");
    }
  }, [router]);

  if (!ready || redirecting) return null;

  return <SidecarProvider>{children}</SidecarProvider>;
}
