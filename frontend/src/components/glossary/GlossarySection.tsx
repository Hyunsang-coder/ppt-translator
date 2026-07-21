"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { ArrowRight, BookOpen, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { GlossaryManagerModal } from "@/components/glossary/GlossaryManagerModal";
import {
  GLOSSARY_STORAGE_KEY,
  useGlossaryStore,
} from "@/stores/glossary-store";

interface GlossarySectionProps {
  disabled?: boolean;
}

export function GlossarySection({ disabled = false }: GlossarySectionProps) {
  const glossaries = useGlossaryStore((state) => state.glossaries);
  const activeGlossaryIds = useGlossaryStore((state) => state.activeGlossaryIds);
  const hydrated = useGlossaryStore((state) => state.hydrated);
  const storageWarning = useGlossaryStore((state) => state.storageWarning);
  const hydrate = useGlossaryStore((state) => state.hydrate);
  const syncFromStorage = useGlossaryStore((state) => state.syncFromStorage);
  const dismissStorageWarning = useGlossaryStore((state) => state.dismissStorageWarning);
  const toggleGlossaryActive = useGlossaryStore((state) => state.toggleGlossaryActive);
  const addEntry = useGlossaryStore((state) => state.addEntry);

  const [managerOpen, setManagerOpen] = useState(false);
  const [quickGlossaryId, setQuickGlossaryId] = useState("");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const closeManager = useCallback(() => setManagerOpen(false), []);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key === GLOSSARY_STORAGE_KEY) syncFromStorage(event.newValue);
    };
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, [syncFromStorage]);

  const activeGlossaries = useMemo(() => {
    const byId = new Map(glossaries.map((glossary) => [glossary.id, glossary]));
    return activeGlossaryIds
      .map((id) => byId.get(id))
      .filter((glossary): glossary is NonNullable<typeof glossary> => Boolean(glossary));
  }, [activeGlossaryIds, glossaries]);

  useEffect(() => {
    if (activeGlossaryIds.includes(quickGlossaryId)) return;
    setQuickGlossaryId(activeGlossaryIds[0] ?? "");
  }, [activeGlossaryIds, quickGlossaryId]);

  const handleQuickAdd = () => {
    if (disabled || !quickGlossaryId || !source.trim() || !target.trim()) return;
    try {
      addEntry(quickGlossaryId, source, target);
      setSource("");
      setTarget("");
      toast.success("용어를 추가했습니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "추가에 실패했습니다.");
    }
  };

  const handleUnlink = (glossaryId: string) => {
    try {
      toggleGlossaryActive(glossaryId);
      toast.message("이번 번역에서 용어집을 제외했습니다. 저장된 용어는 유지됩니다.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "용어집을 제외하지 못했습니다.");
    }
  };

  return (
    <>
      <div className="space-y-3 rounded-xl border border-border p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <Label className="text-sm font-medium">용어집</Label>
            <p className="mt-0.5 text-xs text-muted-foreground">
              저장된 용어집을 선택하거나 원문→번역 용어를 직접 추가합니다.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-8 shrink-0 gap-1.5"
            disabled={disabled}
            onClick={() => setManagerOpen(true)}
          >
            <BookOpen className="h-3.5 w-3.5" />
            관리
          </Button>
        </div>

        {storageWarning && (
          <div className="flex items-start gap-2 rounded-md border border-warning/40 bg-warning/10 px-2.5 py-2 text-xs text-warning">
            <span className="min-w-0 flex-1">{storageWarning}</span>
            <button type="button" onClick={dismissStorageWarning} aria-label="저장소 경고 닫기">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        )}

        {!hydrated ? (
          <div className="rounded-md border border-border px-3 py-3 text-center text-xs text-muted-foreground">
            저장된 용어집을 불러오는 중입니다.
          </div>
        ) : activeGlossaries.length > 0 ? (
          <div className="space-y-1">
            {activeGlossaries.map((glossary, index) => (
              <div key={glossary.id} className="group flex items-center gap-2 rounded-md border border-border bg-muted/40 px-2.5 py-2">
                <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[9px] font-semibold text-primary">
                  {index + 1}
                </span>
                <BookOpen className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate text-sm">
                  {glossary.name}
                  <span className="ml-1.5 text-xs text-muted-foreground">{glossary.entries.length}개</span>
                </span>
                <button
                  type="button"
                  className="text-muted-foreground opacity-0 hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100 disabled:opacity-50"
                  disabled={disabled}
                  onClick={() => handleUnlink(glossary.id)}
                  aria-label={`${glossary.name}을 이번 번역에서 제외`}
                  title="이번 번역에서 제외 (라이브러리는 유지)"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <button
            type="button"
            disabled={disabled}
            onClick={() => setManagerOpen(true)}
            className="w-full rounded-md border border-dashed border-border px-3 py-3 text-center text-xs text-muted-foreground hover:border-primary/50 hover:text-foreground disabled:opacity-50"
          >
            이번 번역에 사용할 용어집을 선택하거나 새로 만드세요
          </button>
        )}

        {activeGlossaries.length > 0 && (
          <div className="space-y-2 rounded-lg border border-border bg-muted/20 p-2.5">
            <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Plus className="h-3.5 w-3.5" />
              빠른 용어 추가
            </div>
            {activeGlossaries.length > 1 && (
              <select
                value={quickGlossaryId}
                onChange={(event) => setQuickGlossaryId(event.target.value)}
                disabled={disabled}
                className="h-8 w-full rounded-md border border-border bg-background px-2 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                aria-label="용어를 추가할 용어집"
              >
                {activeGlossaries.map((glossary) => (
                  <option key={glossary.id} value={glossary.id}>{glossary.name}</option>
                ))}
              </select>
            )}
            <div className="flex flex-wrap items-end gap-2">
              <div className="min-w-[100px] flex-1 space-y-1">
                <Label htmlFor="glossary-quick-source" className="text-xs text-muted-foreground">원문</Label>
                <Input id="glossary-quick-source" value={source} onChange={(event) => setSource(event.target.value)} disabled={disabled} className="h-8 text-sm" placeholder="Source" />
              </div>
              <ArrowRight className="mb-2 h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="min-w-[100px] flex-1 space-y-1">
                <Label htmlFor="glossary-quick-target" className="text-xs text-muted-foreground">번역</Label>
                <Input
                  id="glossary-quick-target"
                  value={target}
                  onChange={(event) => setTarget(event.target.value)}
                  disabled={disabled}
                  className="h-8 text-sm"
                  placeholder="Target"
                  onKeyDown={(event) => {
                    if (event.key === "Enter") handleQuickAdd();
                  }}
                />
              </div>
              <Button type="button" size="sm" className="h-8 gap-1" disabled={disabled || !quickGlossaryId || !source.trim() || !target.trim()} onClick={handleQuickAdd}>
                <Plus className="h-3.5 w-3.5" />
                추가
              </Button>
            </div>
          </div>
        )}
      </div>

      <GlossaryManagerModal open={managerOpen} onClose={closeManager} disabled={disabled} />
    </>
  );
}
