"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { Update } from "@tauri-apps/plugin-updater";
import { isTauri } from "@/lib/api-base";
import { SidecarProvider } from "@/components/sidecar-provider";
import { UpdateModal } from "@/components/ui/update-modal";
import { useAutoUpdate } from "@/hooks/useAutoUpdate";

const LOCAL_WEB_HOSTS = new Set(["localhost", "127.0.0.1", "0.0.0.0", "::1"]);

function isHostedWebBuild(): boolean {
  if (typeof window === "undefined" || isTauri()) return false;
  return !LOCAL_WEB_HOSTS.has(window.location.hostname);
}

/**
 * Owns the auto-update lifecycle for the desktop app: starts the startup check
 * (inside the hook) and shows the modal when an update is available — either
 * from that check or from a manual check elsewhere via the `app:update-found`
 * event. Renders nothing outside Tauri.
 */
function AutoUpdateGate() {
  const {
    available,
    update,
    downloading,
    progress,
    error,
    downloadAndInstall,
    cancelDownload,
    skipVersion,
    dismissUpdate,
    setManualUpdate,
  } = useAutoUpdate();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (available && update) setOpen(true);
  }, [available, update]);

  // 설정 화면 등에서 수동으로 확인한 결과를 같은 모달로 받는다.
  useEffect(() => {
    const handler = (e: Event) => {
      const found = (e as CustomEvent<Update>).detail;
      if (found) {
        setManualUpdate(found);
        setOpen(true);
      }
    };
    window.addEventListener("app:update-found", handler);
    return () => window.removeEventListener("app:update-found", handler);
  }, [setManualUpdate]);

  return (
    <UpdateModal
      isOpen={open}
      version={update?.version ?? ""}
      releaseNotes={update?.body ?? undefined}
      downloading={downloading}
      progress={progress}
      error={error}
      onUpdate={downloadAndInstall}
      onCancel={cancelDownload}
      onSkipVersion={() => {
        if (update?.version) skipVersion(update.version);
        setOpen(false);
      }}
      onDismiss={() => {
        dismissUpdate();
        setOpen(false);
      }}
    />
  );
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

  return (
    <SidecarProvider>
      {isTauri() && <AutoUpdateGate />}
      {children}
    </SidecarProvider>
  );
}
