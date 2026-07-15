"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  ArrowRight,
  BookOpen,
  FileSpreadsheet,
  FileText,
  Pencil,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiClient } from "@/lib/api-client";
import { parseGlossaryCsv, type GlossaryEntry } from "@/lib/glossary";
import { useGlossaryStore } from "@/stores/glossary-store";

interface GlossaryManagerModalProps {
  open: boolean;
  onClose: () => void;
  disabled?: boolean;
}

export function GlossaryManagerModal({
  open,
  onClose,
  disabled = false,
}: GlossaryManagerModalProps) {
  const glossaries = useGlossaryStore((s) => s.glossaries);
  const activeGlossaryId = useGlossaryStore((s) => s.activeGlossaryId);
  const ensureDefaultGlossary = useGlossaryStore((s) => s.ensureDefaultGlossary);
  const setActiveGlossaryId = useGlossaryStore((s) => s.setActiveGlossaryId);
  const createGlossary = useGlossaryStore((s) => s.createGlossary);
  const renameGlossary = useGlossaryStore((s) => s.renameGlossary);
  const deleteGlossary = useGlossaryStore((s) => s.deleteGlossary);
  const addEntry = useGlossaryStore((s) => s.addEntry);
  const updateEntry = useGlossaryStore((s) => s.updateEntry);
  const deleteEntry = useGlossaryStore((s) => s.deleteEntry);
  const importEntries = useGlossaryStore((s) => s.importEntries);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [notes, setNotes] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);

  const csvInputRef = useRef<HTMLInputElement>(null);
  const excelInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const id = ensureDefaultGlossary();
    setSelectedId(activeGlossaryId ?? id);
    setShowNew(false);
    setRenaming(false);
    setSearchQuery("");
    resetEntryForm();
  }, [open, ensureDefaultGlossary, activeGlossaryId]);

  const selected = glossaries.find((g) => g.id === selectedId) ?? null;
  const entries = selected?.entries ?? [];

  const visibleEntries = useMemo(() => {
    const q = searchQuery.trim().toLocaleLowerCase();
    if (!q) return entries;
    return entries.filter(
      (e) =>
        e.source.toLocaleLowerCase().includes(q) ||
        e.target.toLocaleLowerCase().includes(q) ||
        (e.notes?.toLocaleLowerCase().includes(q) ?? false)
    );
  }, [entries, searchQuery]);

  if (!open) return null;

  function resetEntryForm() {
    setSource("");
    setTarget("");
    setNotes("");
    setEditingId(null);
  }

  function selectGlossary(id: string) {
    setSelectedId(id);
    setActiveGlossaryId(id);
    setRenaming(false);
    resetEntryForm();
  }

  function handleCreateGlossary() {
    if (disabled || !newName.trim()) return;
    const id = createGlossary(newName);
    setNewName("");
    setShowNew(false);
    selectGlossary(id);
    toast.success("용어집을 만들었습니다. 이번 번역에 사용됩니다.");
  }

  function handleRename() {
    if (disabled || !selected || !nameDraft.trim()) return;
    renameGlossary(selected.id, nameDraft);
    setRenaming(false);
    toast.success("이름을 변경했습니다.");
  }

  function handleDeleteGlossary() {
    if (disabled || !selected) return;
    if (glossaries.length <= 1) {
      toast.error("마지막 용어집은 삭제할 수 없습니다.");
      return;
    }
    if (!window.confirm(`「${selected.name}」용어집과 ${selected.entries.length}개 용어를 삭제할까요?`)) {
      return;
    }
    deleteGlossary(selected.id);
    toast.success("용어집을 삭제했습니다.");
  }

  function handleSaveEntry() {
    if (disabled || !selected) return;
    try {
      if (editingId) {
        updateEntry(selected.id, editingId, { source, target, notes });
        toast.success("용어를 수정했습니다.");
      } else {
        addEntry(selected.id, source, target, notes);
        toast.success("용어를 추가했습니다.");
      }
      resetEntryForm();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "저장에 실패했습니다.");
    }
  }

  function startEdit(entry: GlossaryEntry) {
    setEditingId(entry.id);
    setSource(entry.source);
    setTarget(entry.target);
    setNotes(entry.notes ?? "");
  }

  async function handleCsvFile(file: File | null) {
    if (!file || !selected || disabled) return;
    setImporting(true);
    try {
      const text = await file.text();
      const parsed = parseGlossaryCsv(text);
      if (parsed.length === 0) {
        toast.error("유효한 용어를 찾지 못했습니다.");
        return;
      }
      const { inserted, updated } = importEntries(selected.id, parsed);
      toast.success(`가져오기 완료: ${inserted}개 추가, ${updated}개 갱신`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "CSV 가져오기에 실패했습니다.");
    } finally {
      setImporting(false);
      if (csvInputRef.current) csvInputRef.current.value = "";
    }
  }

  async function handleExcelFile(file: File | null) {
    if (!file || !selected || disabled) return;
    setImporting(true);
    try {
      const { entries: parsed } = await apiClient.parseGlossaryFile(file);
      if (parsed.length === 0) {
        toast.error("유효한 용어를 찾지 못했습니다.");
        return;
      }
      const { inserted, updated } = importEntries(selected.id, parsed);
      toast.success(`가져오기 완료: ${inserted}개 추가, ${updated}개 갱신`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Excel 가져오기에 실패했습니다.");
    } finally {
      setImporting(false);
      if (excelInputRef.current) excelInputRef.current.value = "";
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="glossary-manager-title"
      onClick={onClose}
    >
      <div
        className="flex h-[min(720px,90vh)] w-full max-w-4xl overflow-hidden rounded-xl border border-border bg-background shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Library sidebar */}
        <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-muted/30">
          <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-3">
            <h2 id="glossary-manager-title" className="text-sm font-semibold">
              용어집
            </h2>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 px-2"
              disabled={disabled}
              onClick={() => setShowNew(true)}
            >
              <Plus className="h-3.5 w-3.5" />
            </Button>
          </div>

          {showNew && (
            <div className="flex gap-1 border-b border-border p-2">
              <Input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="새 용어집 이름"
                className="h-8 text-sm"
                disabled={disabled}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreateGlossary();
                }}
                autoFocus
              />
              <Button
                type="button"
                size="sm"
                className="h-8"
                disabled={disabled || !newName.trim()}
                onClick={handleCreateGlossary}
              >
                추가
              </Button>
            </div>
          )}

          <ul className="flex-1 overflow-y-auto p-2 space-y-1">
            {glossaries.map((g) => {
              const active = g.id === selectedId;
              return (
                <li key={g.id}>
                  <button
                    type="button"
                    onClick={() => selectGlossary(g.id)}
                    className={`w-full rounded-md px-2.5 py-2 text-left text-sm transition-colors ${
                      active
                        ? "bg-primary/10 text-foreground font-medium"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                  >
                    <span className="block truncate">{g.name}</span>
                    <span className="text-[11px] opacity-70">{g.entries.length}개 용어</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </aside>

        {/* Detail */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
            <div className="min-w-0 flex-1">
              {renaming && selected ? (
                <div className="flex gap-2">
                  <Input
                    value={nameDraft}
                    onChange={(e) => setNameDraft(e.target.value)}
                    className="h-8 max-w-xs text-sm"
                    disabled={disabled}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRename();
                      if (e.key === "Escape") setRenaming(false);
                    }}
                    autoFocus
                  />
                  <Button type="button" size="sm" className="h-8" onClick={handleRename}>
                    저장
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <BookOpen className="h-4 w-4 shrink-0 text-muted-foreground" />
                  <h3 className="truncate text-sm font-semibold">
                    {selected?.name ?? "용어집 없음"}
                  </h3>
                  {selected && (
                    <button
                      type="button"
                      className="text-muted-foreground hover:text-foreground"
                      disabled={disabled}
                      onClick={() => {
                        setNameDraft(selected.name);
                        setRenaming(true);
                      }}
                      aria-label="이름 변경"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              )}
              <p className="mt-1 text-xs text-muted-foreground">
                선택한 용어집이 이번 번역에 사용됩니다. Excel/CSV로 가져올 수 있습니다.
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              {selected && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-8 text-destructive hover:text-destructive"
                  disabled={disabled || glossaries.length <= 1}
                  onClick={handleDeleteGlossary}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
              <Button type="button" variant="ghost" size="sm" className="h-8" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {selected && (
            <>
              <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
                <input
                  ref={csvInputRef}
                  type="file"
                  accept=".csv,text/csv"
                  className="hidden"
                  disabled={disabled || importing}
                  onChange={(e) => void handleCsvFile(e.target.files?.[0] ?? null)}
                />
                <input
                  ref={excelInputRef}
                  type="file"
                  accept=".xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
                  className="hidden"
                  disabled={disabled || importing}
                  onChange={(e) => void handleExcelFile(e.target.files?.[0] ?? null)}
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5"
                  disabled={disabled || importing}
                  onClick={() => csvInputRef.current?.click()}
                >
                  <FileText className="h-3.5 w-3.5" />
                  CSV
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5"
                  disabled={disabled || importing}
                  onClick={() => excelInputRef.current?.click()}
                >
                  <FileSpreadsheet className="h-3.5 w-3.5" />
                  Excel
                </Button>
                <div className="relative ml-auto min-w-[160px] max-w-[220px] flex-1">
                  <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="검색"
                    className="h-8 pl-7 text-sm"
                  />
                </div>
              </div>

              <div className="space-y-2 border-b border-border px-4 py-3">
                <div className="flex flex-wrap items-end gap-2">
                  <div className="min-w-[120px] flex-1 space-y-1">
                    <Label className="text-xs text-muted-foreground">원문</Label>
                    <Input
                      value={source}
                      onChange={(e) => setSource(e.target.value)}
                      disabled={disabled}
                      className="h-8 text-sm"
                      placeholder="Source"
                    />
                  </div>
                  <ArrowRight className="mb-2 h-4 w-4 shrink-0 text-muted-foreground" />
                  <div className="min-w-[120px] flex-1 space-y-1">
                    <Label className="text-xs text-muted-foreground">번역</Label>
                    <Input
                      value={target}
                      onChange={(e) => setTarget(e.target.value)}
                      disabled={disabled}
                      className="h-8 text-sm"
                      placeholder="Target"
                      onKeyDown={(e) => {
                        if (e.key === "Enter") handleSaveEntry();
                      }}
                    />
                  </div>
                  <div className="min-w-[100px] flex-1 space-y-1">
                    <Label className="text-xs text-muted-foreground">메모 (선택)</Label>
                    <Input
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      disabled={disabled}
                      className="h-8 text-sm"
                      placeholder="Notes"
                    />
                  </div>
                  <Button
                    type="button"
                    size="sm"
                    className="h-8"
                    disabled={disabled || !source.trim() || !target.trim()}
                    onClick={handleSaveEntry}
                  >
                    {editingId ? "수정" : "추가"}
                  </Button>
                  {editingId && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-8"
                      onClick={resetEntryForm}
                    >
                      취소
                    </Button>
                  )}
                </div>
              </div>

              <ul className="flex-1 overflow-y-auto px-2 py-2">
                {visibleEntries.length === 0 ? (
                  <li className="px-2 py-8 text-center text-sm text-muted-foreground">
                    {entries.length === 0
                      ? "용어가 없습니다. 위 폼이나 CSV/Excel로 추가하세요."
                      : "검색 결과가 없습니다."}
                  </li>
                ) : (
                  visibleEntries.map((entry) => (
                    <li
                      key={entry.id}
                      className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted/60"
                    >
                      <span className="min-w-0 flex-1 truncate text-sm">
                        <span className="font-medium">{entry.source}</span>
                        <span className="mx-1.5 text-muted-foreground">→</span>
                        <span>{entry.target}</span>
                        {entry.notes && (
                          <span className="ml-2 text-xs text-muted-foreground">
                            ({entry.notes})
                          </span>
                        )}
                      </span>
                      <button
                        type="button"
                        className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-foreground"
                        disabled={disabled}
                        onClick={() => startEdit(entry)}
                        aria-label="수정"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive"
                        disabled={disabled}
                        onClick={() => {
                          deleteEntry(selected.id, entry.id);
                          if (editingId === entry.id) resetEntryForm();
                        }}
                        aria-label="삭제"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </li>
                  ))
                )}
              </ul>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
