"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { apiClient } from "@/lib/api-client";
import { glossaryTermKey } from "@/lib/glossary";
import { useGlossaryStore } from "@/stores/glossary-store";
import { Loader2, Plus } from "lucide-react";
import { toast } from "sonner";

interface GlossaryPaneProps {
  jobId: string;
  disabled: boolean;
  /** Findings are swept again after a term lands, so the queue must reload. */
  onRegistered: () => Promise<void>;
}

/** The glossary quick-add, moved out of the header into its own column. */
export function GlossaryPane({ jobId, disabled, onRegistered }: GlossaryPaneProps) {
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [busy, setBusy] = useState(false);
  const [glossaryId, setGlossaryId] = useState("");

  const glossaries = useGlossaryStore((s) => s.glossaries);
  const activeGlossaryIds = useGlossaryStore((s) => s.activeGlossaryIds);
  const ensureDefaultGlossary = useGlossaryStore((s) => s.ensureDefaultGlossary);
  const addEntry = useGlossaryStore((s) => s.addEntry);
  const updateEntry = useGlossaryStore((s) => s.updateEntry);
  const deleteEntry = useGlossaryStore((s) => s.deleteEntry);
  const setActiveGlossaryIds = useGlossaryStore((s) => s.setActiveGlossaryIds);

  const activeGlossaries = useMemo(() => {
    const byId = new Map(glossaries.map((glossary) => [glossary.id, glossary]));
    return activeGlossaryIds
      .map((id) => byId.get(id))
      .filter((glossary): glossary is NonNullable<typeof glossary> => Boolean(glossary));
  }, [activeGlossaryIds, glossaries]);

  useEffect(() => {
    if (activeGlossaryIds.includes(glossaryId)) return;
    setGlossaryId(activeGlossaryIds[0] ?? "");
  }, [activeGlossaryIds, glossaryId]);

  const entryCount = activeGlossaries.reduce(
    (total, glossary) => total + glossary.entries.length,
    0
  );

  const register = async () => {
    const src = source.trim();
    const tgt = target.trim();
    if (!src || !tgt || busy) return;
    setBusy(true);
    const previousActiveIds = [...useGlossaryStore.getState().activeGlossaryIds];
    let localRollback: (() => void) | null = null;
    try {
      // Validate and persist locally first. If the job update fails, compensate
      // the local mutation so the current review and future jobs do not diverge.
      const targetGlossaryId = glossaryId || ensureDefaultGlossary();
      const glossary = useGlossaryStore
        .getState()
        .glossaries.find((item) => item.id === targetGlossaryId);
      const existing = glossary?.entries.find((entry) => (
        glossaryTermKey(entry.source) === glossaryTermKey(src)
      ));
      if (existing) {
        const previous = { ...existing };
        updateEntry(targetGlossaryId, existing.id, { source: src, target: tgt });
        localRollback = () => {
          updateEntry(targetGlossaryId, existing.id, previous);
          setActiveGlossaryIds(previousActiveIds);
        };
        if (!previousActiveIds.includes(targetGlossaryId)) {
          setActiveGlossaryIds([...previousActiveIds, targetGlossaryId]);
        }
      } else {
        const result = addEntry(targetGlossaryId, src, tgt);
        localRollback = () => {
          deleteEntry(targetGlossaryId, result.entry.id);
          setActiveGlossaryIds(previousActiveIds);
        };
      }
      await apiClient.updateJobGlossary(jobId, { [src]: tgt });
      setSource("");
      setTarget("");
      await onRegistered();
      toast.success("용어집에 추가했습니다. 재번역 시 적용됩니다.");
    } catch (err) {
      if (localRollback) {
        try {
          localRollback();
        } catch {
          toast.error("현재 작업 반영에 실패했고 로컬 용어집 복구도 완료하지 못했습니다.");
          setBusy(false);
          return;
        }
      }
      toast.error(err instanceof Error ? err.message : "용어집 추가에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <aside className="flex w-[264px] shrink-0 flex-col border-l border-border bg-card">
      <div className="border-b border-border px-4 pb-3.5 pt-4">
        <p className="text-[13px] font-bold">이 덱의 용어집</p>
        <p className="truncate text-xs text-muted-foreground">
          {activeGlossaries.length > 0
            ? `${activeGlossaries.map((glossary) => glossary.name).join(" · ")} · ${entryCount}개`
            : "선택된 용어집 없음"}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 text-xs text-muted-foreground">
        용어를 추가하면 이 작업의 검출을 다시 계산합니다.
      </div>

      <div className="border-t border-border px-4 pb-4 pt-3">
        <p className="mb-2 text-xs font-semibold text-muted-foreground">
          고른 문구를 용어집에 추가
        </p>
        {activeGlossaries.length > 1 && (
          <select
            value={glossaryId}
            onChange={(event) => setGlossaryId(event.target.value)}
            disabled={busy || disabled}
            aria-label="저장할 용어집"
            className="mb-2 h-8 w-full rounded-md border border-border bg-background px-2 text-[13px] outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            {activeGlossaries.map((glossary) => (
              <option key={glossary.id} value={glossary.id}>{glossary.name}</option>
            ))}
          </select>
        )}
        <Input
          value={source}
          onChange={(event) => setSource(event.target.value)}
          disabled={busy || disabled}
          className="h-8 text-[13px]"
          placeholder="원문"
          aria-label="용어 원문"
        />
        <Input
          value={target}
          onChange={(event) => setTarget(event.target.value)}
          disabled={busy || disabled}
          className="mt-1.5 h-8 text-[13px]"
          placeholder="번역"
          aria-label="용어 번역"
          onKeyDown={(event) => {
            if (event.key === "Enter") void register();
          }}
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="mt-2 h-8 w-full gap-1"
          disabled={busy || disabled || !source.trim() || !target.trim()}
          onClick={() => void register()}
        >
          {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Plus className="size-3.5" />}
          추가
        </Button>
        <p className="mt-2 text-[11px] leading-normal text-muted-foreground">
          라이브러리와 이 작업에 동시에 반영 · 다음 번역부터 자동 적용됩니다.
        </p>
      </div>
    </aside>
  );
}
