"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { ArrowRight, BookOpen, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { GlossaryManagerModal } from "@/components/glossary/GlossaryManagerModal";
import { useGlossaryStore } from "@/stores/glossary-store";

interface GlossarySectionProps {
  disabled?: boolean;
}

export function GlossarySection({ disabled = false }: GlossarySectionProps) {
  const glossaries = useGlossaryStore((s) => s.glossaries);
  const activeGlossaryId = useGlossaryStore((s) => s.activeGlossaryId);
  const ensureDefaultGlossary = useGlossaryStore((s) => s.ensureDefaultGlossary);
  const setActiveGlossaryId = useGlossaryStore((s) => s.setActiveGlossaryId);
  const addEntry = useGlossaryStore((s) => s.addEntry);
  const getActiveGlossary = useGlossaryStore((s) => s.getActiveGlossary);

  const [managerOpen, setManagerOpen] = useState(false);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");

  useEffect(() => {
    ensureDefaultGlossary();
  }, [ensureDefaultGlossary]);

  const active = getActiveGlossary();
  const entryCount = active?.entries.length ?? 0;

  const handleQuickAdd = () => {
    if (disabled || !source.trim() || !target.trim()) return;
    try {
      const id = ensureDefaultGlossary();
      addEntry(id, source, target);
      setSource("");
      setTarget("");
      toast.success("용어를 추가했습니다.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "추가에 실패했습니다.");
    }
  };

  const handleClearActive = () => {
    // Clearing means "don't use glossary for this job" without deleting the library.
    setActiveGlossaryId(null);
    toast.message("이번 번역에는 용어집을 사용하지 않습니다.");
  };

  const handleUseAgain = () => {
    if (glossaries[0]) {
      setActiveGlossaryId(glossaries[0].id);
    } else {
      ensureDefaultGlossary();
    }
  };

  return (
    <>
      <div className="space-y-3 rounded-xl border border-border p-4">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <Label className="text-sm font-medium">용어집</Label>
            <p className="mt-0.5 text-xs text-muted-foreground">
              원문→번역 용어를 추가하거나 Excel/CSV로 가져옵니다.
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

        {active && activeGlossaryId ? (
          <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-2.5 py-2">
            <BookOpen className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
            <span className="min-w-0 flex-1 truncate text-sm">
              {active.name}
              <span className="ml-1.5 text-xs text-muted-foreground">{entryCount}개</span>
            </span>
            <button
              type="button"
              className="text-muted-foreground hover:text-foreground disabled:opacity-50"
              disabled={disabled}
              onClick={handleClearActive}
              aria-label="이번 번역에서 용어집 제외"
              title="이번 번역에서 제외 (라이브러리는 유지)"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : (
          <button
            type="button"
            disabled={disabled}
            onClick={() => {
              handleUseAgain();
              setManagerOpen(true);
            }}
            className="w-full rounded-md border border-dashed border-border px-3 py-3 text-center text-xs text-muted-foreground hover:border-primary/50 hover:text-foreground disabled:opacity-50"
          >
            용어집을 선택하거나 관리에서 추가하세요
          </button>
        )}

        {activeGlossaryId && (
          <div className="flex flex-wrap items-end gap-2">
            <div className="min-w-[100px] flex-1 space-y-1">
              <Label htmlFor="glossary-quick-source" className="text-xs text-muted-foreground">
                원문
              </Label>
              <Input
                id="glossary-quick-source"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                disabled={disabled}
                className="h-8 text-sm"
                placeholder="Source"
              />
            </div>
            <ArrowRight className="mb-2 h-4 w-4 shrink-0 text-muted-foreground" />
            <div className="min-w-[100px] flex-1 space-y-1">
              <Label htmlFor="glossary-quick-target" className="text-xs text-muted-foreground">
                번역
              </Label>
              <Input
                id="glossary-quick-target"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                disabled={disabled}
                className="h-8 text-sm"
                placeholder="Target"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleQuickAdd();
                }}
              />
            </div>
            <Button
              type="button"
              size="sm"
              className="h-8 gap-1"
              disabled={disabled || !source.trim() || !target.trim()}
              onClick={handleQuickAdd}
            >
              <Plus className="h-3.5 w-3.5" />
              추가
            </Button>
          </div>
        )}
      </div>

      <GlossaryManagerModal
        open={managerOpen}
        onClose={() => setManagerOpen(false)}
        disabled={disabled}
      />
    </>
  );
}
